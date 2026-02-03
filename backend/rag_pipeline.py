import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", override=True)

def format_docs(docs) -> str:
    # keep context small and clean
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", "")
        text = d.page_content.strip()
        parts.append(f"[source={src}, page={page}]\n{text}")
    return "\n\n".join(parts)

def format_history(history) -> str:
    if not history:
        return ""
    history = history[-6:]  # last 3 turns
    lines = []
    for m in history:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)
def load_hybrid_retriever():
    """
    Loads Pinecone Hybrid retriever (dense+BM25).
    Make sure bm25_values.json exists on backend (created during ingestion).
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"].strip())
    index_name = os.environ["PINECONE_INDEX"]  # should be your hybrid index name
    index = pc.Index(index_name)

    bm25_path = Path(__file__).parent / "bm25_values.json"
    bm25 = BM25Encoder().load(str(bm25_path))

    alpha = float(os.getenv("HYBRID_ALPHA", "0.75"))
    retrieve_k = int(os.getenv("RETRIEVE_K", "10"))  # candidates for reranker

    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        top_k=retrieve_k,
        alpha=alpha
    )

def load_reranker():
    """
    CrossEncoder reranker. Small + fast: good default.
    You can switch to "BAAI/bge-reranker-base" if you want stronger (slower).
    """
    model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder(model_name)

def rerank_docs(question: str, docs, reranker, top_n: int):
    if not docs:
        return []
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:top_n]]


def load_chat_model():
    repo_id = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    assert token, "HF token not found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=token,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "384")),
        do_sample=False,
        repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.1")),
    )
    return ChatHuggingFace(llm=llm)

def answer_question(
    question: str,
    history=None,
    retriever=None,
    reranker=None,
    chat_model=None
) -> Dict:
    """
    Uses:
    1) Hybrid retrieve (top K candidates)
    2) Rerank (choose best N)
    3) LLM answers using ONLY context
    """
    assert retriever is not None and reranker is not None and chat_model is not None, "Pipeline not initialized"

    # 1) Retrieve candidates
    candidates = retriever.invoke(question)

    if not candidates:
        return {"answer": "I don't know based on the documents.", "sources": []}

    # 2) Rerank
    top_n = int(os.getenv("RERANK_TOP_N", "5"))
    docs = rerank_docs(question, candidates, reranker, top_n=top_n)

    if not docs:
        return {"answer": "I don't know based on the documents.", "sources": []}

    # 3) Build context + history
    context = format_docs(docs)
    history_text = format_history(history)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful RAG assistant. Use ONLY the provided context to answer.\n"
         "If the answer is not in the context, say: 'I don't know based on the documents.'\n"
         "Chat history is for continuity, not as a factual source unless supported by context."),
        ("user",
         "Chat history:\n{history}\n\n"
         "Question: {question}\n\n"
         "Context:\n{context}")
    ])

    chain = prompt | chat_model | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context, "history": history_text})

    return {"answer": answer, "sources": [d.metadata for d in docs]}

#
