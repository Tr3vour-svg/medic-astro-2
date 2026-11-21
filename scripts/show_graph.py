"""Render and sanity-check the compiled graph. Run from the project root:

    python -m scripts.show_graph

Outputs (in ./docs):
  graph.md   Mermaid diagram (renders on GitHub, in mermaid.live, or a VS Code Mermaid preview)
  graph.png  PNG image (needs internet: LangGraph renders it through the mermaid.ink service)
"""
import os
from collections import defaultdict, deque

from core.graph import app

OUT_DIR = "docs"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    g = app.get_graph()

    # ── 1. Mermaid source (always works, no network) ────────────────────────
    md_path = os.path.join(OUT_DIR, "graph.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MEDIC ASTRO graph\n\n```mermaid\n" + g.draw_mermaid() + "\n```\n")
    print(f"✅ Mermaid diagram written to {md_path}")

    # ── 2. PNG image (needs internet) ───────────────────────────────────────
    try:
        png_path = os.path.join(OUT_DIR, "graph.png")
        with open(png_path, "wb") as f:
            f.write(g.draw_mermaid_png())
        print(f"✅ PNG written to {png_path}")
    except Exception as e:
        print(f"⚠️ PNG rendering skipped ({type(e).__name__}: {str(e)[:100]}). Use {md_path} instead.")

    # ── 3. ASCII diagram in the terminal (needs: pip install grandalf) ──────
    try:
        print("\n── ASCII graph ─────────────────────────────────────────────")
        print(g.draw_ascii())
    except ImportError:
        print("ℹ️ For the terminal diagram run: pip install grandalf")
    except Exception as e:
        print(f"⚠️ ASCII rendering failed: {e}")

    # ── 4. Adjacency list ───────────────────────────────────────────────────
    edges = defaultdict(list)
    for edge in g.edges:
        label = "  (conditional)" if getattr(edge, "conditional", False) else ""
        edges[edge.source].append(f"{edge.target}{label}")

    print("\n── Adjacency (node → next) ─────────────────────────────────")
    for src in sorted(edges):
        print(f"  {src:<20} → {', '.join(sorted(edges[src]))}")

    # ── 5. Reachability check ───────────────────────────────────────────────
    graph_edges = defaultdict(set)
    for edge in g.edges:
        graph_edges[edge.source].add(edge.target)

    seen, queue = {"__start__"}, deque(["__start__"])
    while queue:
        for nxt in graph_edges[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    all_nodes = set(g.nodes)
    unreachable = sorted(all_nodes - seen)
    dead_ends = sorted(n for n in all_nodes if n != "__end__" and not graph_edges[n])

    print("\n── Checks ──────────────────────────────────────────────────")
    print(f"  Nodes (incl. __start__/__end__): {len(all_nodes)}")
    print(f"  {'✅' if not unreachable else '❌'} Unreachable from start: {unreachable or 'none'}")
    print(f"  {'✅' if not dead_ends else '❌'} Dead ends (no outgoing edge): {dead_ends or 'none'}")
    print(f"  {'✅' if '__end__' in seen else '❌'} END is reachable")


if __name__ == "__main__":
    main()