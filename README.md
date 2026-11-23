---
title: MEDIC ASTRO
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Medic-Astro

A multi-agent hospital policy assistant built with LangGraph, Pinecone, and Gradio.

## Project Structure

```
medic-astro/
│
├── .env                    # API keys (never commit this file)
├── .gitignore              # Files excluded from Git (.env, __pycache__, venv)
├── README.md               # Project documentation
├── app.py                  # Entry point (launches the Gradio UI)
├── requirements.txt        # Python dependencies
│
└── core/
    ├── __init__.py         # Makes 'core' a Python package
    ├── config.py           # Environment variables, LLMs, embeddings, vector store
    ├── state.py            # LangGraph state and reducers
    ├── tools.py            # Turso, Pinecone, and Tavily tools
    ├── guardrails.py       # Input and output security checks
    ├── nodes.py            # All agent nodes
    ├── graph.py            # Graph assembly and compilation
    └── ui.py               # Gradio interface
```

## Setup

1. Clone the repository and move into the project folder:
   ```bash
   cd medic-astro
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac / Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=your-key-here
   PINECONE_API_KEY=your-key-here
   DEEPSEEK_API_KEY=your-key-here
   ```

5. Run the app from the project root:
   ```bash
   python app.py
   ```

## Module Overview

| File | Responsibility |
|------|----------------|
| `app.py` | Starts the application |
| `core/config.py` | Loads environment variables and defines models and the vector store |
| `core/state.py` | Defines the shared state passed between graph nodes |
| `core/tools.py` | External tools the agents can call |
| `core/guardrails.py` | Validates user input and checks model output |
| `core/nodes.py` | Supervisor, retrieval, synthesis, and review nodes |
| `core/graph.py` | Connects the nodes into a LangGraph workflow |
| `core/ui.py` | Builds the Gradio chat interface |
