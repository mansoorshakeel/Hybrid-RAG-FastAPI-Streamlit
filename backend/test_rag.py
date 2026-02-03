import os
from rag_pipeline import (
    load_hybrid_retriever,
    load_reranker,
    load_chat_model,
    answer_question
)

def main():
    # 1) Init once (like production)
    retriever = load_hybrid_retriever()
    reranker = load_reranker()
    chat_model = load_chat_model()

    # 2) Run a test question that MUST exist in your docs
    q = "What is the difference between TCP and UDP?"
    out = answer_question(
        q,
        history=[],
        retriever=retriever,
        reranker=reranker,
        chat_model=chat_model
    )

    print("\nANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(s)

if __name__ == "__main__":
    main()
