"""Central settings, overridable through environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

VIDEOS_DIR = Path(os.getenv("RAG_VIDEOS_DIR", BASE_DIR / "videos"))
AUDIOS_DIR = Path(os.getenv("RAG_AUDIOS_DIR", BASE_DIR / "audios"))
JSONS_DIR = Path(os.getenv("RAG_JSONS_DIR", BASE_DIR / "jsons"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "index"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2")
WHISPER_MODEL = os.getenv("RAG_WHISPER_MODEL", "large-v2")

# Texts sent to the embedding endpoint per request.
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "64"))
# Consecutive Whisper segments merged into one retrievable chunk.
# Whisper segments are only a few seconds long; merging them gives the
# retriever more context per hit and shrinks the index.
CHUNK_WINDOW = int(os.getenv("RAG_CHUNK_WINDOW", "4"))
# Segments shared between neighbouring chunks.
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "1"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
REQUEST_TIMEOUT = float(os.getenv("RAG_REQUEST_TIMEOUT", "300"))
# Parallel ffmpeg processes in video_to_mp3.
FFMPEG_WORKERS = int(os.getenv("RAG_FFMPEG_WORKERS", str(os.cpu_count() or 2)))

COURSE_DESCRIPTION = os.getenv(
    "RAG_COURSE_DESCRIPTION", "I am teaching web development in my Sigma web development course."
)
