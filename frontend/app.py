# frontend/app.py
import os
import requests
import streamlit as st

# Example: set in Streamlit Cloud / env
# BACKEND_URL="https://your-backend-domain.com"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot (Hybrid + Reranker)")

# ---- Session state for chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}

# ---- Render chat history ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- Input ----
user_text = st.chat_input("Ask from your documents...")

if user_text:
    # 1) Store user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) Send last 6 messages (3 turns) as history
    history_to_send = st.session_state.messages[-6:]

    # 3) Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": user_text, "history": history_to_send},
                    timeout=120
                )
                r.raise_for_status()
                data = r.json()

                answer = data.get("answer", "")
                sources = data.get("sources", [])

                st.markdown(answer)

                # Show sources (optional)
                with st.expander("Sources"):
                    # pretty print: source + page if present
                    for s in sources:
                        src = s.get("source", "")
                        page = s.get("page", "")
                        score = s.get("score", None)
                        if score is not None:
                            st.write(f"- {src} | page: {page} | score: {score}")
                        else:
                            st.write(f"- {src} | page: {page}")

            except requests.exceptions.RequestException as e:
                st.error(f"Backend error: {e}")
                answer = "Sorry, I couldn't reach the backend API."

    # 4) Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- Sidebar utilities ----
with st.sidebar:
    st.header("Settings")
    st.caption("Backend URL (set via BACKEND_URL env var)")
    st.code(BACKEND_URL)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
