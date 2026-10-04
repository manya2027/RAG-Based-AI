"""On-disk vector index: a normalised float32 matrix plus chunk metadata.

Layout of INDEX_DIR:
  embeddings.npy  - (n_chunks, dim) float32, rows L2-normalised
  chunks.jsonl    - one metadata record per row, same order
  manifest.json   - source json file -> mtime, used for incremental rebuilds
"""
import json
from pathlib import Path

import numpy as np

import config


def normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    def __init__(self, embeddings, chunks, manifest=None):
        self.embeddings = embeddings
        self.chunks = chunks
        self.manifest = manifest or {}

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 0), dtype=np.float32), [], {})

    @classmethod
    def load(cls, index_dir=config.INDEX_DIR, mmap=True):
        index_dir = Path(index_dir)
        emb_path = index_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"No index at {index_dir}. Run preprocess_json.py first.")
        # Memory-map so several API workers share one copy of the matrix via the page cache.
        embeddings = np.load(emb_path, mmap_mode="r" if mmap else None)
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return cls(embeddings, chunks, manifest)

    def save(self, index_dir=config.INDEX_DIR):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        # Write to temp files then rename, so readers never see a half-written index.
        tmp_emb = index_dir / "embeddings.tmp.npy"
        np.save(tmp_emb, np.ascontiguousarray(self.embeddings, dtype=np.float32))
        tmp_chunks = index_dir / "chunks.jsonl.tmp"
        with open(tmp_chunks, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        tmp_manifest = index_dir / "manifest.json.tmp"
        tmp_manifest.write_text(json.dumps(self.manifest, indent=2))
        tmp_emb.replace(index_dir / "embeddings.npy")
        tmp_chunks.replace(index_dir / "chunks.jsonl")
        tmp_manifest.replace(index_dir / "manifest.json")

    def __len__(self):
        return len(self.chunks)

    def search(self, query_embedding, top_k=config.TOP_K):
        if len(self) == 0:
            return []
        query = normalize(query_embedding).reshape(-1)
        scores = self.embeddings @ query
        top_k = min(top_k, len(scores))
        # argpartition is O(n); only the top_k survivors get fully sorted.
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [{**self.chunks[i], "score": float(scores[i])} for i in idx]
