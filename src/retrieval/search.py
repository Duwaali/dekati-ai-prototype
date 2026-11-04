#!/usr/bin/env python3
# Hybrid search: combines meaning search, keyword search and the Finnish index into one ranking.


import os
import re
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()

INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
SQLITE_PATH = INDEX_DIR / "chunks.sqlite"
FAISS_PATH = INDEX_DIR / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MULTI_MODEL_NAME = "intfloat/multilingual-e5-small"
MULTI_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index_multilingual"

CANDIDATES = 50
RRF_K = 60

STOPWORDS = {
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "by",
    "can", "do", "does", "for", "from", "get", "give", "has", "have", "how",
    "i", "in", "is", "it", "its", "kind", "list", "make", "makes", "me",
    "much", "my", "of", "on", "or", "our", "report", "tell", "that", "the",
    "their", "there", "these", "this", "to", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "with", "you", "your",
    "ja", "on", "ei", "se", "mikä", "mitä", "kuinka", "miten", "onko",
}


# The search engine: meaning search, keyword search and the Finnish index together.
class HybridSearcher:
    # Load the indexes and the models.
    def __init__(self, index_dir=INDEX_DIR):
        import faiss
        from sentence_transformers import SentenceTransformer

        sqlite_path = Path(index_dir) / "chunks.sqlite"
        faiss_path = Path(index_dir) / "faiss.index"
        if not sqlite_path.exists() or not faiss_path.exists():
            raise FileNotFoundError("Index not built yet. Run src/retrieval/build_index.py")

        faiss.omp_set_num_threads(1)
        self.index = faiss.read_index(str(faiss_path))
        self.db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.model = SentenceTransformer(MODEL_NAME)

        self.multi_index = self.multi_model = None
        multi_path = Path(index_dir).parent / "index_multilingual" / "faiss.index"
        if multi_path.exists():
            self.multi_index = faiss.read_index(str(multi_path))
            self.multi_model = SentenceTransformer(MULTI_MODEL_NAME)

        embedded = self.db.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
        if embedded != self.index.ntotal:
            raise RuntimeError(f"SQLite has {embedded} embedded rows but FAISS has {self.index.ntotal}")

    # Find chunks with a similar meaning to the question.
    def semantic(self, question, k=CANDIDATES):
        vector = self.model.encode([question], normalize_embeddings=True,
                                   convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    # Find Finnish chunks that match the question, in any language.
    def semantic_multilingual(self, question, k=CANDIDATES):
        if self.multi_index is None:
            return []
        vector = self.multi_model.encode(["query: " + question], normalize_embeddings=True,
                                         convert_to_numpy=True).astype("float32")
        scores, ids = self.multi_index.search(vector, min(k, self.multi_index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    # Turn the question into a safe keyword query.
    @staticmethod
    def keyword_query(question):
        terms = []
        for word in re.findall(r"\w+", question.lower()):
            if word in STOPWORDS or len(word) < 2 or word in terms:
                continue
            terms.append(word)
        return " OR ".join(f'"{t}"' for t in terms)

    # Find chunks that contain the question's words.
    def keyword(self, question, k=CANDIDATES):
        query = self.keyword_query(question)
        if not query:
            return []
        rows = self.db.execute(
            "SELECT rowid, bm25(chunks_fts, 1.0, 0.5) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (query, k),
        ).fetchall()
        return [(int(r[0]), -float(r[1])) for r in rows]

    # Get one chunk from the database.
    def fetch(self, row_id):
        return dict(self.db.execute("SELECT * FROM chunks WHERE row_id = ?",
                                    (row_id,)).fetchone())

    MIN_CONTEXT_WORDS = 60
    MAX_EXTRA = 4
    MAX_HITS_PER_DOC = 2
    RICH_SOURCES = 2
    RICH_MIN_WORDS = 250
    RICH_MAX_EXTRA = 10
    MAX_CONTEXT_WORDS = 300

    # Get the chunks around a found chunk in the same document.
    def neighbours(self, chunk, window, min_words=None, max_extra=None, forward=False):
        if window <= 0:
            return [chunk]
        row = chunk["row_id"]
        min_words = min_words or self.MIN_CONTEXT_WORDS
        reach = window + (max_extra or self.MAX_EXTRA)
        rows = [dict(r) for r in self.db.execute(
            "SELECT * FROM chunks WHERE doc_id = ? AND row_id BETWEEN ? AND ? ORDER BY row_id",
            (chunk["doc_id"], row - reach, row + reach))]
        position = next(i for i, r in enumerate(rows) if r["row_id"] == row)

        # Count the words in some chunks.
        def words(part):
            return sum(len((r["text"] or "").split()) for r in part)

        before = after = window
        part = rows[max(0, position - before): position + after + 1]
        while words(part) < min_words and max(before, after) < reach:
            after += 1
            if not forward:
                before += 1
            part = rows[max(0, position - before): position + after + 1]
            if position + after + 1 > len(rows) and (forward or position - before <= 0):
                break
        return part

    # Choose which chunks fit in the context: the found chunks first (best first), then the
    # chunks closest to them, until the word limit is reached; then put them in document order.
    # Before this, chunks were taken in document order, so an earlier, weaker hit could fill the
    # limit and push the best chunk out of the context.
    def fit_context(self, parts, hit_rows, forward=False):
        def priority(row):
            dist = [abs(row - h) for h in hit_rows]
            nearest = dist.index(min(dist))
            before = row < hit_rows[nearest]
            return (min(dist), nearest, before if forward else not before, row)
        kept, words = [], 0
        for row in sorted(parts, key=priority):
            n = len((parts[row]["text"] or "").split())
            if words and words + n > self.MAX_CONTEXT_WORDS:
                continue
            kept.append(row); words += n
        return [parts[r] for r in sorted(kept)]

    # List every place where the same text appears.
    def copies(self, chunk, limit=20):
        canonical = chunk["duplicate_of"] or chunk["chunk_id"]
        where = "chunk_id = ? OR duplicate_of = ?"
        total = self.db.execute(f"SELECT COUNT(*) FROM chunks WHERE {where}",
                                (canonical, canonical)).fetchone()[0]
        rows = self.db.execute(
            f"SELECT chunk_id, doc_id, source_type, source_file, parent_source_file, title, "
            f"location FROM chunks WHERE {where} ORDER BY row_id LIMIT ?",
            (canonical, canonical, limit)).fetchall()
        return total, [dict(r) for r in rows]

    # The older search behaviour, kept to compare against.
    def _legacy_results(self, ranked, detail, top_k, context_window):
        results, seen_texts, boilerplate = [], set(), []
        for row_id, score in ranked:
            chunk = self.fetch(row_id)
            canonical = chunk["duplicate_of"] or chunk["chunk_id"]
            if canonical in seen_texts:
                continue
            seen_texts.add(canonical)
            chunk.update(detail[row_id], rrf_score=round(score, 5))
            if chunk["is_boilerplate"]:
                boilerplate.append(chunk)
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break
        if len(results) < top_k:
            results.extend(boilerplate[: top_k - len(results)])
        for chunk in results:
            chunk["copies_total"], chunk["copies"] = self.copies(chunk)
            parts = [dict(r) for r in self.db.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND section_id = ? "
                "AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
                (chunk["doc_id"], chunk["section_id"],
                 chunk["chunk_index"] - context_window, chunk["chunk_index"] + context_window))]
            chunk["context_text"] = " ".join(p["text"] for p in parts)
            chunk["context_chunks"] = [p["chunk_id"] for p in parts]
        return results

    # Combine all searches and build the sources for the answer.
    def search(self, question, top_k=6, context_window=1, legacy=False):
        semantic = self.semantic(question)
        keyword = self.keyword(question)
        multilingual = self.semantic_multilingual(question)

        fused = {}
        detail = {}
        for method, hits in (("semantic", semantic), ("keyword", keyword),
                             ("multilingual", multilingual)):
            for rank, (row_id, score) in enumerate(hits, start=1):
                fused[row_id] = fused.get(row_id, 0.0) + 1.0 / (RRF_K + rank)
                detail.setdefault(row_id, {})[f"{method}_rank"] = rank
                detail[row_id][f"{method}_score"] = round(score, 4)

        ranked = sorted(fused.items(), key=lambda x: -x[1])
        if legacy:
            return self._legacy_results(ranked, detail, top_k, context_window)

        docs, order, seen_texts, weak = {}, [], set(), []
        for row_id, score in ranked:
            chunk = self.fetch(row_id)
            canonical = chunk["duplicate_of"] or chunk["chunk_id"]
            if canonical in seen_texts:
                continue
            seen_texts.add(canonical)
            chunk.update(detail[row_id], rrf_score=round(score, 5))
            if chunk["is_boilerplate"] or is_table_of_contents(chunk["text"]):
                weak.append(chunk)
                continue
            if chunk["doc_id"] in docs:
                if len(docs[chunk["doc_id"]]["hits"]) < self.MAX_HITS_PER_DOC:
                    docs[chunk["doc_id"]]["hits"].append(chunk)
                continue
            if len(order) >= top_k:
                continue
            docs[chunk["doc_id"]] = {"best": chunk, "hits": [chunk]}
            order.append(chunk["doc_id"])
        for chunk in weak:
            if len(order) >= top_k:
                break
            if chunk["doc_id"] not in docs:
                docs[chunk["doc_id"]] = {"best": chunk, "hits": [chunk]}
                order.append(chunk["doc_id"])

        results = []
        for rank, doc_id in enumerate(order):
            entry = docs[doc_id]
            rich = rank < self.RICH_SOURCES
            chunk = entry["best"]
            chunk["copies_total"], chunk["copies"] = self.copies(chunk)
            parts = {}
            for hit in entry["hits"]:
                extra = ({"min_words": self.RICH_MIN_WORDS, "max_extra": self.RICH_MAX_EXTRA,
                          "forward": True} if rich else {})
                for p in self.neighbours(hit, context_window, **extra):
                    parts[p["row_id"]] = p
            kept = self.fit_context(parts, [h["row_id"] for h in entry["hits"]], forward=rich)
            chunk["context_text"] = " ".join(p["text"] or "" for p in kept)
            chunk["context_chunks"] = [p["chunk_id"] for p in kept]
            chunk["hits_in_document"] = len(entry["hits"])
            results.append(chunk)
        return results


_TOC_LINE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S.*?\s\d{1,3}\s*$")


# Check if a passage is just a table of contents.
def is_table_of_contents(text):
    text = text or ""
    head = text.strip()[:40].lower()
    if head.startswith(("sisällysluettelo", "table of contents", "contents")):
        return True
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return False
    return sum(bool(_TOC_LINE.match(l)) for l in lines) / len(lines) >= 0.5


# Choose the name to show for a source.
def source_name(chunk):
    return (chunk.get("parent_source_file") or chunk.get("title")
            or chunk.get("source_file") or chunk.get("doc_id"))


# Search from the command line and show the sources.
def main():
    args = sys.argv[1:]
    if "--root" in args:
        i = args.index("--root")
        del args[i:i + 2]
    if not args:
        print('usage: python src/retrieval/search.py "question"')
        sys.exit(1)
    searcher = HybridSearcher()
    for n, r in enumerate(searcher.search(args[0]), start=1):
        print(f"[{n}] rrf {r['rrf_score']:.4f}  "
              f"semantic #{r.get('semantic_rank', '-')}  keyword #{r.get('keyword_rank', '-')}  "
              f"{r['source_type']}  {source_name(r)}  |  {r.get('location') or '-'}")


if __name__ == "__main__":
    main()
