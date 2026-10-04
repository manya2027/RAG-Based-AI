# How to use this RAG AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by running `python video_to_mp3.py`.
Videos are converted in parallel (`RAG_FFMPEG_WORKERS`) and ones already converted are skipped.

## Step 3 - Convert mp3 to json
Convert all the mp3 files to json by running `python mp3_to_json.py` (needs `pip install openai-whisper`).
Audio files that already have a json are skipped, so adding new videos only transcribes the new ones.

## Step 4 - Convert the json files to Vectors
Run `python preprocess_json.py` to build the vector index in `index/`.
- Consecutive Whisper segments are merged into overlapping chunks (`RAG_CHUNK_WINDOW`, `RAG_CHUNK_OVERLAP`),
  which gives each hit more context and makes the index ~3x smaller.
- Embeddings are requested in batches (`RAG_EMBED_BATCH_SIZE`) with automatic retries.
- Runs are incremental: only new or modified json files are embedded, deleted ones are dropped.
  Use `--rebuild` to start over (done automatically if the embedding model or chunk settings change).

## Step 5 - Ask questions
Interactive CLI (loads the index once, then answers as many questions as you like):

    python process_incoming.py

HTTP API for many users at once:

    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

- `POST /ask` `{"question": "...", "top_k": 5}` -> answer + source chunks with timestamps
- `POST /search` -> retrieval only, no LLM call
- `GET /health`

The index is memory-mapped, so all workers share one copy of the vectors.

## Running with Docker
After building the index (step 4):

    docker compose up -d
    docker compose exec ollama ollama pull bge-m3
    docker compose exec ollama ollama pull llama3.2

## Configuration
All settings live in `config.py` and can be overridden with environment variables, e.g.
`OLLAMA_URL`, `RAG_EMBED_MODEL`, `RAG_LLM_MODEL`, `RAG_WHISPER_MODEL`, `RAG_TOP_K`, `RAG_INDEX_DIR`.
To scale the LLM side further, point `OLLAMA_URL` at a load balancer in front of several Ollama servers.
