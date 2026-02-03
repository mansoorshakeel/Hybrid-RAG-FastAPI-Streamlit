# RAG Frontend (Streamlit)

This is the Streamlit UI for the RAG chatbot. It calls a FastAPI backend over HTTP.

## Local Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

set BACKEND_URL=http://127.0.0.1:8000
streamlit run streamlit_app.py
