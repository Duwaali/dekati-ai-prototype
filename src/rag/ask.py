#!/usr/bin/env python3
# Answers a question: finds the best passages and lets the local language model write a sourced answer.


import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from search import HybridSearcher, source_name

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:3b"
TOP_K = 3
CONTEXT_WINDOW = 1
NOT_FOUND = "I could not find this in the Dekati documents."

SYSTEM_PROMPT = f"""You answer questions for Dekati Ltd employees using ONLY the numbered sources you are given.

How to answer:
- The sources may be written in Finnish or in English. Read them in their own language and write your answer in the language of the question.
- Unless the question asks for a report or overview, answer briefly and to the point: a few sentences at most, or a short numbered list for steps. No introduction, no summary of the sources.
- Read every source carefully before deciding that the answer is missing. The answer may be in any of them.
- If a source contains steps, a list, a procedure or specifications, give them completely and in the original order.
- Put the source number after each fact, like [1] or [2][3].
- Copy numbers, units and product names exactly.
- If the sources only partly answer the question, give what they contain and say briefly what is missing.
- Only if none of the sources is related to the question, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge. Do not repeat these instructions or the source headings."""


REPORT_TOP_K = 5
REPORT_WORDS = ("report", "overview", "summary", "summarise", "summarize", "tell me everything",
                "raportti", "yhteenveto", "katsaus")
REPORT_FORMAT = """Write a short report with these parts, using only the sources:
Relationship: who or what this is in relation to Dekati, in one or two sentences.
Key facts: three to six bullet points (products, orders, quotes, projects, dates, places).
Recent activity: the latest dated events in the sources, if any.
Not found: what the sources do not say that a reader would expect.
Put the source number after every fact."""


ADVISOR_TOP_K = 8
ADVISOR_MAX_TOKENS = 700   # stops very long answers, which took over 10 minutes on the laptop
ADVISOR_FORMAT = """Answer in two parts.
From the documents: the relevant facts from the sources, as bullet points, with the source number after every fact.
Suggestions: recommendations or possible solutions that follow from these facts. Start each suggestion with the source numbers it is based on, like "Based on [2][5]:". Do not suggest anything that the sources do not support.
If the task needs a count, an average, a ranking or a check of all documents, do not guess a number: say that it cannot be calculated from these sources and what data would be needed."""


# Check if the question asks for a report or overview.
def is_report_question(question):
    q = question.lower()
    return any(word in q for word in REPORT_WORDS)


# Decide how many sources to use: more for report questions and in advisor mode.
def sources_for_question(question, mode="facts"):
    if mode == "advisor":
        return ADVISOR_TOP_K
    return REPORT_TOP_K if is_report_question(question) else TOP_K


# Put the numbered sources and the question together for the model.
def build_prompt(question, results, mode="facts"):
    blocks = []
    for n, r in enumerate(results, start=1):
        location = r.get("location") or "-"
        blocks.append(f"[{n}] ({source_name(r)}, {location})\n{r['context_text']}")
    sources = "\n\n".join(blocks)
    if mode == "advisor":
        task = f"\n\n{ADVISOR_FORMAT}"
    else:
        task = f"\n\n{REPORT_FORMAT}" if is_report_question(question) else ""
    return f"{sources}\n\nQuestion: {question}{task}"


KEEP_ALIVE = "30m"


# Prepare the request that is sent to the local model.
def ollama_request(model, prompt, stream, max_tokens=None):
    options = {"temperature": 0, "num_ctx": 8192}
    if max_tokens:
        options["num_predict"] = max_tokens
    return {
        "model": model,
        "stream": stream,
        "keep_alive": KEEP_ALIVE,
        "options": options,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }


# Ask the local model and wait for the full answer.
def call_ollama(model, prompt, timeout=600, max_tokens=None):
    response = requests.post(OLLAMA_URL, json=ollama_request(model, prompt, False, max_tokens), timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# Load the model into memory at start, so the first question is faster.
def preload_model(model=DEFAULT_MODEL):
    requests.post("http://localhost:11434/api/generate",
                  json={"model": model, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": 8192}},
                  timeout=300).raise_for_status()


# Search, then send the answer piece by piece as it is written.
def stream_answer(question, searcher, model=DEFAULT_MODEL, top_k=None, timeout=600):
    t0 = time.time()
    results = searcher.search(question, top_k=top_k or sources_for_question(question),
                              context_window=CONTEXT_WINDOW)
    yield {"type": "sources", "results": results, "retrieval_seconds": round(time.time() - t0, 2)}
    yield from generate_stream(question, results, model=model, timeout=timeout)


# Let the model write the answer and send it piece by piece.
def generate_stream(question, results, model=DEFAULT_MODEL, timeout=600, mode="facts"):
    t1 = time.time()
    if not results:
        yield {"type": "token", "text": NOT_FOUND}
        yield {"type": "done", "answer": NOT_FOUND, "grounding": grounding_check(NOT_FOUND, 0),
               "generation_seconds": 0.0, "model": model}
        return
    parts = []
    max_tokens = ADVISOR_MAX_TOKENS if mode == "advisor" else None
    with requests.post(OLLAMA_URL, json=ollama_request(model, build_prompt(question, results, mode), True, max_tokens),
                       stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = (data.get("message") or {}).get("content", "")
            if piece:
                parts.append(piece)
                yield {"type": "token", "text": piece}
            if data.get("done"):
                break
    answer = "".join(parts).strip()
    yield {"type": "done", "answer": answer, "grounding": grounding_check(answer, len(results)),
           "generation_seconds": round(time.time() - t1, 2), "model": model}


# Check that the answer only cites sources that were really given.
def grounding_check(answer, n_sources):
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    invalid = [n for n in cited if n < 1 or n > n_sources]
    not_found = NOT_FOUND.lower().rstrip(".") in answer.lower()
    return {
        "cited_sources": cited,
        "invalid_citations": invalid,
        "has_citation": bool(cited),
        "said_not_found": not_found,
        "ok": not invalid and (bool(cited) or not_found),
    }


# Search for sources and get one complete answer.
def answer_question(question, searcher, model=DEFAULT_MODEL, top_k=None, mode="facts"):
    t0 = time.time()
    results = searcher.search(question, top_k=top_k or sources_for_question(question, mode),
                              context_window=CONTEXT_WINDOW)
    t1 = time.time()
    if not results:
        return {"answer": NOT_FOUND, "sources": [], "grounding": grounding_check(NOT_FOUND, 0),
                "retrieval_seconds": round(t1 - t0, 2), "generation_seconds": 0.0,
                "model": model, "results": []}
    answer = call_ollama(model, build_prompt(question, results, mode),
                         max_tokens=ADVISOR_MAX_TOKENS if mode == "advisor" else None)
    t2 = time.time()
    return {
        "answer": answer,
        "sources": [{
            "n": n,
            "document": source_name(r),
            "type": r["source_type"],
            "location": r.get("location"),
            "chunk_id": r["chunk_id"],
            "semantic_rank": r.get("semantic_rank"),
            "keyword_rank": r.get("keyword_rank"),
            "also_in": [{"document": source_name(c), "location": c.get("location")}
                        for c in r.get("copies", []) if c["chunk_id"] != r["chunk_id"]],
            "copies_total": r.get("copies_total", 1),
        } for n, r in enumerate(results, start=1)],
        "grounding": grounding_check(answer, len(results)),
        "retrieval_seconds": round(t1 - t0, 2),
        "generation_seconds": round(t2 - t1, 2),
        "model": model,
        "results": results,
    }


# Answer a question typed on the command line.
def main():
    args = sys.argv[1:]
    model = DEFAULT_MODEL
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
    if not args:
        print('usage: python src/rag/ask.py [--model NAME] "question"')
        sys.exit(1)

    print("Loading search index...")
    searcher = HybridSearcher()
    result = answer_question(args[0], searcher, model=model)

    print()
    print("ANSWER")
    print("-" * 70)
    print(result["answer"])
    print()
    print("SOURCES")
    print("-" * 70)
    for s in result["sources"]:
        print(f"[{s['n']}] {s['document']}  |  {s['location'] or '-'}  ({s['type']})")
        for other in s["also_in"][:5]:
            print(f"      same text also in: {other['document']}  |  {other['location'] or '-'}")
        if s["copies_total"] - 1 > 5:
            print(f"      ... and {s['copies_total'] - 6} more places")
    g = result["grounding"]
    print()
    print(f"grounding check : {'PASS' if g['ok'] else 'CHECK MANUALLY'} "
          f"(cited {g['cited_sources'] or 'none'})")
    print(f"time            : search {result['retrieval_seconds']} s, "
          f"answer {result['generation_seconds']} s ({result['model']})")


if __name__ == "__main__":
    main()
