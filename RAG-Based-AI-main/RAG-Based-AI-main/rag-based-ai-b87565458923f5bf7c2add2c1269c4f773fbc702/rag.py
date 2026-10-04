"""Retrieval + prompt construction shared by the CLI and the API."""
import json

import config
from ollama_client import create_embedding, inference
from vector_index import VectorIndex


def build_prompt(query, results):
    records = [
        {k: r[k] for k in ("title", "number", "start", "end", "text")} for r in results
    ]
    return f'''{config.COURSE_DESCRIPTION} Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{json.dumps(records, ensure_ascii=False)}
---------------------------------
"{query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''


class RAG:
    def __init__(self, index=None):
        self.index = index if index is not None else VectorIndex.load()

    def retrieve(self, query, top_k=config.TOP_K):
        question_embedding = create_embedding([query])[0]
        return self.index.search(question_embedding, top_k)

    def answer(self, query, top_k=config.TOP_K):
        results = self.retrieve(query, top_k)
        prompt = build_prompt(query, results)
        response = inference(prompt)["response"]
        return {"answer": response, "prompt": prompt, "sources": results}
