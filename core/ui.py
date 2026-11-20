"""Gradio interface: streaming chat, voice input, and human-in-the-loop review panel."""
import uuid

import gradio as gr
from langgraph.types import Command

from core.config import CONFIG, message_text, whisper_client
from core.graph import app
from core.state import build_initial_state

PHASE_LABELS = {
    "evaluating": "🔍 Evaluating your query…",
    "compiling":  "🧩 Compiling sources…",
    "generating": "✍️ Generating answer…",
}


# ═════════════════════════════════════════════════════════════════════════════
# Session helpers
# ═════════════════════════════════════════════════════════════════════════════
def _new_thread_id(user_id: str) -> str:
    return f"{user_id}-{uuid.uuid4().hex[:8]}"


def _graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _status_line(agent_status: dict) -> str:
    if not agent_status:
        return "_no agent activity yet_"
    return "  ·  ".join(f"`{k}` → {v}" for k, v in agent_status.items())


# ═════════════════════════════════════════════════════════════════════════════
# Core streaming runner (shared by new messages and review resumes)
# ═════════════════════════════════════════════════════════════════════════════
def _run_graph(graph_input, config, history, thread_state, prefix: str = ""):
    """
    Stream one graph run into the chat. Yields tuples matching the UI outputs:
    (chatbot, msg_box, thread_state, status_box, review_box)
    """
    status_line = "⏳ Working…"
    partial = ""
    review_draft = ""
    paused = False

    try:
        for mode, chunk in app.stream(
            graph_input, config, stream_mode=["custom", "messages", "updates"]
        ):
            if mode == "custom" and isinstance(chunk, dict):
                phase = chunk.get("phase", "")
                if phase == "evaluating":
                    partial = ""  # a new synthesis pass starts — drop tokens of earlier drafts
                label = PHASE_LABELS.get(phase, phase)
                status_line = f"{label} {chunk.get('detail', '')}".strip()
                yield history, "", thread_state, status_line, ""

            elif mode == "messages":
                msg_chunk, meta = chunk
                if (meta or {}).get("langgraph_node") == "medic_nova":
                    token = message_text(msg_chunk)
                    if token:
                        partial += token
                        history[-1] = {"role": "assistant", "content": f"{prefix}{partial}▍"}
                        yield history, "", thread_state, status_line, ""

            elif mode == "updates":
                for node_name, update in chunk.items():
                    if node_name == "__interrupt__":
                        interrupts = update if isinstance(update, (list, tuple)) else [update]
                        if interrupts:
                            payload = getattr(interrupts[0], "value", interrupts[0]) or {}
                            review_draft = payload.get("draft", "")
                            paused = True
                            history[-1] = {
                                "role": "assistant",
                                "content": f"⏸️ *Awaiting human review:*\n\n{review_draft}",
                            }
                            status_line = "⏸️ Awaiting human review"
                            yield history, "", thread_state, status_line, review_draft
                        continue

                    if node_name.startswith("__"):
                        continue

                    if isinstance(update, dict):
                        st = update.get("agent_status") or {}
                        if st:
                            key = next(iter(st))
                            status_line = f"✓ {key}: {st[key]}"
                            yield history, "", thread_state, status_line, ""

    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"⚠️ System error: {e}"}
        yield history, "", thread_state, f"_error: {str(e)[:120]}_", ""
        return

    if paused:
        yield history, "", thread_state, "⏸️ Awaiting human review", review_draft
        return

    # Show the authoritative, guardrail-sanitized answer rather than the raw streamed tokens.
    # This also covers runs that never reach MEDIC NOVA (e.g. blocked by the input guardrail).
    values = app.get_state(config).values or {}
    final_text = values.get("final_output") or partial or "(no response generated)"
    history[-1] = {"role": "assistant", "content": f"{prefix}{final_text}"}
    yield history, "", thread_state, _status_line(values.get("agent_status") or {}), ""


# ═════════════════════════════════════════════════════════════════════════════
# Handlers
# ═════════════════════════════════════════════════════════════════════════════
def respond(message, history, user_id, clearance, department, thread_state):
    message = (message or "").strip()
    if not message:
        yield history, "", thread_state, "_empty input_", ""
        return

    if not thread_state:
        thread_state = _new_thread_id(user_id)

    config = _graph_config(thread_state)
    initial_state = build_initial_state(message, user_id, clearance, department)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, "", thread_state, "⏳ Starting…", ""
    yield from _run_graph(initial_state, config, history, thread_state)


def transcribe_audio(audio_path, history, user_id, clearance, department, thread_state):
    if not audio_path:
        yield history, "", thread_state, "_No audio detected_", ""
        return

    try:
        with open(audio_path, "rb") as audio_file:
            transcript = whisper_client.audio.transcriptions.create(
                model=CONFIG["transcription_model"],
                file=audio_file,
            )
        user_message = transcript.text
    except Exception as e:
        yield history, "", thread_state, f"_Audio error: {e}_", ""
        return

    yield from respond(user_message, history, user_id, clearance, department, thread_state)


def _resume_review(payload: dict, thread_state, history, prefix: str = ""):
    if not thread_state:
        yield [], "", thread_state, "_no active review_", ""
        return
    if not history:
        history = [{"role": "assistant", "content": ""}]

    yield from _run_graph(
        Command(resume=payload), _graph_config(thread_state), history, thread_state, prefix=prefix
    )


def approve_review(edited_text, thread_state, history):
    payload = {"action": "approve", "edited_draft": edited_text}
    yield from _resume_review(payload, thread_state, history, prefix="✅ *Reviewed response:*\n\n")


def reject_review(edited_text, thread_state, history):
    payload = {"action": "reject", "edited_draft": edited_text}
    yield from _resume_review(payload, thread_state, history)


def reset_session():
    return [], "", None, "_new session_", ""


# ═════════════════════════════════════════════════════════════════════════════
# Theme & CSS (dark)
# ═════════════════════════════════════════════════════════════════════════════
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0f1115",
    body_background_fill_dark="#0f1115",
    block_background_fill="#1a1d24",
    block_border_color="#2d333b",
    block_title_text_color="#8b949e",
    body_text_color="#c9d1d9",
    body_text_color_subdued="#8b949e",
    button_primary_background_fill="#238636",
    button_primary_background_fill_hover="#2ea043",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#21262d",
    button_secondary_text_color="#c9d1d9",
    input_background_fill="#0d1117",
    input_border_color="#30363d",
)

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto; padding: 20px; }
footer { display: none !important; }

.message-wrap { border-radius: 8px; }
.message.user { background-color: #2d333b !important; color: #fff !important; }
.message.bot { background-color: #161b22 !important; border: 1px solid #30363d; color: #c9d1d9 !important; }

#status-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    color: #58a6ff;
    background: #0d1117;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #30363d;
    margin-top: 10px;
}

#review-box textarea {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0d1117 !important;
    border: 1px solid #d29922 !important;
}

.header-title { font-weight: 800; letter-spacing: -0.5px; }
"""

HEADER_HTML = """
<div style="text-align: center; margin-bottom: 20px;">
    <h1 class="header-title" style="color: #e6edf3;">🩺 MEDIC ASTRO</h1>
    <p style="color: #8b949e; font-size: 1.1em;">
        Secure Hospital Agent <span style="color: #58a6ff;">|</span>
        Policy <span style="color: #58a6ff;">|</span>
        HR <span style="color: #58a6ff;">|</span>
        Patient <span style="color: #58a6ff;">|</span>
        External Research
    </p>
</div>
"""


# ═════════════════════════════════════════════════════════════════════════════
# Layout
# ═════════════════════════════════════════════════════════════════════════════
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MEDIC ASTRO", theme=CUSTOM_THEME, css=CSS) as demo:
        gr.Markdown(HEADER_HTML)
        thread_state = gr.State(value=None)

        with gr.Row():
            # ── Left: chat & audio ──────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Agent Telemetry", height=550, buttons=["copy"])
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Voice Command (Whisper)",
                    buttons=[],
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type a secure query…",
                        label="Command Line",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    reset_btn = gr.Button("🔄 Clear Session", variant="secondary")
                status_box = gr.Markdown("_System Ready_", elem_id="status-box")

            # ── Right: identity & human-in-the-loop ─────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🔐 Access Control")
                user_id_box = gr.Textbox(
                    label="User ID",
                    value="EMP-000005",
                    info="Padded 6-digit IDs (e.g. EMP-000001 CMO)",
                )
                clearance_box = gr.Dropdown(
                    label="Clearance Level",
                    choices=["Executive", "Department Head", "Physician", "Intern"],
                    value="Intern",
                )
                department_box = gr.Dropdown(
                    label="Department",
                    choices=["HR", "Nursing", "Cardiology", "Radiology", "Oncology", "Administration"],
                    value="Administration",
                )

                gr.Markdown("---")
                gr.Markdown("### 🧑‍⚕️ Human-in-the-Loop")
                review_box = gr.Textbox(
                    label="Pending Action",
                    lines=10,
                    interactive=True,
                    elem_id="review-box",
                    placeholder="Awaiting interrupt... Draft will appear here.",
                )
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve", variant="primary")
                    reject_btn = gr.Button("❌ Reject", variant="stop")

        # ── Wiring ──────────────────────────────────────────────────────
        chat_inputs = [msg_box, chatbot, user_id_box, clearance_box, department_box, thread_state]
        outputs = [chatbot, msg_box, thread_state, status_box, review_box]

        send_btn.click(respond, inputs=chat_inputs, outputs=outputs)
        msg_box.submit(respond, inputs=chat_inputs, outputs=outputs)

        audio_input.stop_recording(
            transcribe_audio,
            inputs=[audio_input, chatbot, user_id_box, clearance_box, department_box, thread_state],
            outputs=outputs,
        )

        approve_btn.click(approve_review, inputs=[review_box, thread_state, chatbot], outputs=outputs)
        reject_btn.click(reject_review, inputs=[review_box, thread_state, chatbot], outputs=outputs)
        reset_btn.click(reset_session, outputs=outputs)

    return demo