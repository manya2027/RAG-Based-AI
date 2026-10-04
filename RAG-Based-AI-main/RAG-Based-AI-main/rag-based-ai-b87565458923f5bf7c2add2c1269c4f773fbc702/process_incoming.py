"""Interactive CLI: ask questions against the index (loaded once per session)."""
from rag import RAG


def main():
    rag = RAG()
    print(f"Loaded {len(rag.index)} chunks. Empty line or Ctrl+C to quit.")
    while True:
        try:
            incoming_query = input("Ask a Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not incoming_query:
            break

        result = rag.answer(incoming_query)
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(result["prompt"])
        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(result["answer"])
        print(result["answer"])


if __name__ == "__main__":
    main()
