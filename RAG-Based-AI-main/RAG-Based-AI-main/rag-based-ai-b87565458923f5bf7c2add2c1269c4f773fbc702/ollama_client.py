"""Thin Ollama client with connection pooling, retries and batched embeddings."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

_session = requests.Session()
_retry = Retry(
    total=4,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"POST"}),
)
_session.mount("http://", HTTPAdapter(max_retries=_retry, pool_maxsize=32))
_session.mount("https://", HTTPAdapter(max_retries=_retry, pool_maxsize=32))


def create_embedding(text_list, batch_size=config.EMBED_BATCH_SIZE):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    embeddings = []
    for start in range(0, len(text_list), batch_size):
        r = _session.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": text_list[start:start + batch_size]},
            timeout=config.REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        embeddings.extend(r.json()["embeddings"])
    return embeddings


def inference(prompt):
    r = _session.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=config.REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
