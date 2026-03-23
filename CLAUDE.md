# CLAUDE.md — AI-Mentor Hub

## Project Purpose
RAG (Retrieval-Augmented Generation) system for AI Engineering students. Lets users query a vectorized library of technical books and receive structured, cited responses.

## Architecture
```
PDFs (data/)
  → PyPDFLoader
  → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=150)
  → HuggingFaceEmbeddings (all-MiniLM-L6-v2, runs LOCAL — no API cost)
  → DocArrayInMemorySearch (in-memory vector store)
  → JSON persistence (data/processed_docs.json)
  → similarity_search (k=4)
  → Gemini 2.5 Flash (google-genai SDK) with structured output
  → RespuestaMentor (Pydantic schema)
  → Streamlit UI (src/app.py)
```

## Key Files
| File | Role |
|------|------|
| `src/ingestion.py` | Load PDFs → chunk → embed → save JSON |
| `src/rag_base.py` | Retrieval + Gemini generation |
| `src/schemas.py` | Pydantic output contract (`RespuestaMentor`, `ReferenciaBibliografica`) |
| `src/app.py` | Streamlit UI (main entry point) |
| `tests/test_rag.py` | Tests for `consultar_mentor()` |
| `tests/test_schemas.py` | Tests for Pydantic schema validation |
| `conftest.py` | Injects `src/` into `sys.path` for all tests |
| `opencode.json` | OpenCode default model configuration |
| `CODE_ANALYSIS.md` | Full code analysis report with issues and fixes |

## How to Run
```bash
# Install dependencies (uses uv)
uv sync

# Run the app (must be inside src/ because ingestion.py uses relative path "data/")
cd src
streamlit run app.py
```

## How to Run Tests
```bash
# From project root — conftest.py handles sys.path automatically
pytest tests/
```

## Environment Variables
- `GOOGLE_API_KEY` — Gemini API key (required). Load from `.env` via `python-dotenv`.
- `.env` is gitignored. Never commit it.

## Current Session Status (2026-03-23)
- ✅ Complete code analysis performed
- ✅ `opencode.json` created with `opencode/minimax-m2.5-free` as default model
- ✅ `CODE_ANALYSIS.md` created documenting all issues found
- ⏳ Pending fixes listed in `TODO.md`

## Important Constraints / Known Issues

### Hardware Compatibility (reason for strict pinning)
- Developed on Intel Gen 3 CPU **without AVX2 support**
- `numpy==1.26.4` (not 2.x) — required for compatibility
- `transformers==4.44.2` (not ≥4.45) — required for compatibility
- `torch==2.2.2` — tested version, do not upgrade lightly
- Do NOT change these versions without testing on the target machine.

### Test Import Issue
- `tests/test_schemas.py` uses `from src.schemas import ...` — this is WRONG
- `tests/test_rag.py` uses `from rag_base import ...` — this is CORRECT
- All test imports should use bare module names (e.g. `from schemas import ...`) because `conftest.py` already adds `src/` to the path.

### vector_db.docs Access (app.py ~line 85)
- `DocArrayInMemorySearch` may not expose a `.docs` attribute reliably
- The current code falls back to `os.listdir()` and then shows a generic message
- Do not add more fallback layers; investigate the actual DocArrayInMemorySearch API if fixing this.

### Relative Paths
- `ingestion.py` uses `"data/"` as a relative path — it must be run from inside `src/` or the CWD must be `src/`
- Streamlit is launched from `src/` for this reason

## Data / Knowledge Base
Books stored as PDFs in `data/` (gitignored):
- AI Engineering — Chip Huyen
- AI Agents in Action — Michael Lanham
- AI Agents and Applications With LangChain, LangGraph, and MCP — Roberto Infante
- Generative AI with LangChain — Dr. Priyanka

Processed vector data stored in `data/processed_docs.json` (gitignored — regenerated on first run).

## Development History Files (kept locally, gitignored)
These files are intentionally ignored but preserved for study:
- `src/app_01_trabajando_con_2_libros_fijo.py` — earlier version of the app
- `src/chromadb_ingestion.py` — Chroma experiment (abandoned due to hardware limits)
- `src/chromadb_rag_base.py` — Chroma experiment
- `src/google_ingestion_rompe_limite_free.py` — Google Embeddings experiment (hit quota)
- `src/analisis_mentor_hub.txt` — analysis notes
- `tests/main_muestra.py` — manual test samples

## Package Manager
`uv` — use `uv sync` to install, `uv add <pkg>` to add dependencies. Do not use `pip install` directly.

## Language Convention
- UI strings and comments: Spanish
- Docstrings and core logic: mix of Spanish/English (do not refactor language unless asked)
- Pydantic field descriptions: English

## Related Documentation
- `CODE_ANALYSIS.md` — Full analysis report with issues categorized by severity
- `TODO.md` — Pending tasks and fixes prioritized by importance
- `PROGRESS.md` — Development history and session log
