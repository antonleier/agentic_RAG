# pip install weave wandb
# wandb login   (paste your API key when prompted)

import weave

# Use YOUR entity. From your quickstart this is "vejaleier-lss".
weave.init("vejaleier-lss/weavehacks-rag")  # 🐝

@weave.op  # 🐝 traces inputs/outputs/latency of this function
def retrieve(query: str) -> list[str]:
    # Dummy "retriever" — pretend these came from a vector DB
    docs = [
        "Weave traces LLM and agent calls.",
        "RAG combines retrieval with generation.",
        "Self-improving agents read their own eval scores.",
    ]
    return [d for d in docs if any(w in d.lower() for w in query.lower().split())]

@weave.op
def generate(query: str, context: list[str]) -> str:
    # Dummy "LLM" — just stitches context together
    joined = " ".join(context) if context else "no relevant context found"
    return f"Answer to '{query}': based on — {joined}"

@weave.op
def rag_pipeline(query: str) -> str:
    context = retrieve(query)
    return generate(query, context)

if __name__ == "__main__":
    print(rag_pipeline("what is weave"))
    print(rag_pipeline("self-improving agents"))