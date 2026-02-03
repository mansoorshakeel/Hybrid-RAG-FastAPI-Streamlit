import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chatbot")
st.title("📚 RAG Chatbot (Pinecone)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask from your documents...")

if user_text:
    # 1) Add user message to session history
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) Build history to send (last 6 messages => last 3 turns)
    history_to_send = st.session_state.messages[-6:]

    # 3) Call backend with message + history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = requests.post(
                f"{BACKEND_URL}/chat",
                json={"message": user_text, "history": history_to_send},
                timeout=120
            )
            r.raise_for_status()
            data = r.json()

            st.markdown(data["answer"])
            with st.expander("Sources"):
                st.write(data.get("sources", []))

    # 4) Save assistant response to session history
    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
