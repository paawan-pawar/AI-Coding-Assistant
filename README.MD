# AI Coding Assistant (Local Ollama + Streamlit)

A lightweight coding assistant you can run locally using **Ollama** (free, local LLMs) with a **Streamlit** UI, plus a simple CLI. It also includes an **optional** “agentic” prototype that uses OpenAI + LangChain/LangGraph (not required for the local workflow).

## What’s included

- **Streamlit app** with tabs for:
  - Chat
  - Code generation
  - Code review
  - Debugging help
  - Refactoring
  - Test generation
  - Documentation generation
  - (Optional) knowledge base indexing + semantic search
- **CLI assistant** (local Ollama)
- **Optional agentic assistant prototype** (OpenAI + LangChain/LangGraph)

## Requirements

- Python 3.10+ recommended
- [Ollama](https://ollama.com/) installed
- An Ollama code model pulled locally (default: `deepseek-coder:6.7b-instruct`)

## Setup

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama (or ensure it’s already running) and pull the default model:

```bash
ollama serve
ollama pull deepseek-coder:6.7b-instruct
```

To use a different model, edit `Config.OLLAMA_MODEL` in `free_coding_assistant.py`.

## Run (Streamlit UI)

Launch the app with Streamlit:

```bash
python -m streamlit run streamlit_app.py
```

Notes:
- Running `python streamlit_app.py` directly is not supported; the script will print the correct Streamlit command.
- Streamlit file watching is disabled via `.streamlit/config.toml` to avoid noisy watcher issues with some ML dependencies.

## Run (CLI)

Run the local CLI assistant:

```bash
python free_coding_assistant.py
```

CLI commands:
- `/help` – show help
- `/file <path>` – add a single file to the knowledge base
- `/index <path>` – index a directory
- `/clear` – clear conversation history
- `/exit` – quit

## Knowledge base (ChromaDB)

The assistant can index code into a persistent ChromaDB store for semantic retrieval during code generation.

- Storage path: `./chroma_db` (see `Config.CHROMA_DB_PATH`)
- Persistent DB file: `chroma_db/chroma.sqlite3`

If you want to reset the knowledge base, stop the app and delete the contents of `chroma_db/`.

Implementation detail:
- The vector store is **lazily initialized**. If Chroma/embeddings fail to import on your machine, the assistant will fall back to a no-op store (knowledge features won’t work, but chat/code features can still run).
- On first use, the embedding model (`all-MiniLM-L6-v2`) may be downloaded by `sentence-transformers`.

## Optional: Agentic assistant (OpenAI + LangChain)

`agentic_coding_assistant.py` is a separate prototype that uses OpenAI + LangChain/LangGraph tools (file IO, command execution, basic workflow graph).

It is **not** wired into the Streamlit app and its dependencies are **not** listed in `requirements.txt`.

To try it:

1) Install extra dependencies (example):

```bash
pip install openai langchain langgraph langchain-openai python-dotenv gitpython watchdog
```

2) Set your API key:

- Windows PowerShell:
	```powershell
	$env:OPENAI_API_KEY = "YOUR_KEY"
	```

3) Run:

```bash
python agentic_coding_assistant.py
```

## Repo layout

- `streamlit_app.py` – Streamlit UI
- `free_coding_assistant.py` – local Ollama assistant + optional Chroma semantic retrieval + CLI
- `agentic_coding_assistant.py` – optional OpenAI/LangChain agentic prototype
- `requirements.txt` – dependencies for the local (Ollama/Streamlit/Chroma) workflow
- `chroma_db/` – persistent vector DB storage
- `code_store/`, `contexts/` – created automatically (reserved for future use)

