"""Build or incrementally update the vector index from the transcript jsons.

Only json files that are new or modified since the last run are embedded;
chunks from deleted files are dropped. Pass --rebuild to start from scratch.
"""
import argparse
import json

import numpy as np

import config
from ollama_client import create_embedding
from vector_index import VectorIndex, normalize


def window_chunks(segments, window=config.CHUNK_WINDOW, overlap=config.CHUNK_OVERLAP):
    """Merge consecutive Whisper segments into larger, overlapping chunks."""
    window = max(1, window)
    step = max(1, window - max(0, overlap))
    merged = []
    for start in range(0, len(segments), step):
        group = segments[start:start + window]
        merged.append({
            "number": group[0]["number"],
            "title": group[0]["title"],
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(s["text"].strip() for s in group),
        })
        if start + window >= len(segments):
            break
    return merged


def index_settings():
    return {
        "embed_model": config.EMBED_MODEL,
        "chunk_window": config.CHUNK_WINDOW,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="ignore the existing index")
    args = parser.parse_args()

    json_files = sorted(p for p in config.JSONS_DIR.glob("*.json") if p.is_file())
    current = {p.name: p.stat().st_mtime for p in json_files}

    try:
        index = VectorIndex.load(mmap=False)
    except FileNotFoundError:
        index = VectorIndex.empty()
    if args.rebuild or index.manifest.get("settings") != index_settings():
        index = VectorIndex.empty()

    previous = index.manifest.get("files", {})
    unchanged = {name for name, mtime in current.items() if previous.get(name) == mtime}
    to_embed = [p for p in json_files if p.name not in unchanged]

    keep = [i for i, c in enumerate(index.chunks) if c["source"] in unchanged]
    chunks = [index.chunks[i] for i in keep]
    blocks = [np.asarray(index.embeddings[keep], dtype=np.float32)] if keep else []

    for path in to_embed:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)
        file_chunks = window_chunks(content["chunks"])
        if not file_chunks:
            continue
        print(f"Creating Embeddings for {path.name} ({len(file_chunks)} chunks)")
        embeddings = create_embedding([c["text"] for c in file_chunks])
        for chunk in file_chunks:
            chunk["source"] = path.name
        chunks.extend(file_chunks)
        blocks.append(normalize(embeddings))

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    embeddings = np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=np.float32)
    VectorIndex(embeddings, chunks, {"settings": index_settings(), "files": current}).save()
    removed = len(set(previous) - set(current))
    print(f"Index ready: {len(chunks)} chunks from {len(current)} files "
          f"({len(to_embed)} embedded, {len(unchanged)} reused, {removed} removed).")


if __name__ == "__main__":
    main()
