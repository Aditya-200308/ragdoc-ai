# ============================================================
# FILE: src/retriever.py
# PURPOSE: Stores chunks in ChromaDB and retrieves the most
#          relevant ones for a given question.
#          Also implements RE-RANKING to improve result quality.
#
# TWO PHASES:
#   INDEXING:   store all document chunks (done once per document)
#   RETRIEVAL:  given a user question, find the most relevant chunks
#
# THE RE-RANKING BONUS:
#   ChromaDB gives us the top-K chunks by embedding similarity.
#   But embedding similarity is "approximate" — it can return
#   chunks that are nearby in vector space but not actually relevant.
#   Re-ranking uses a more precise model (cross-encoder) to re-score
#   and reorder those top-K chunks for higher accuracy.
# ============================================================


import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 📦 IMPORT: chromadb — our vector database
import chromadb

# 📦 IMPORT: CrossEncoder from sentence-transformers
from sentence_transformers import CrossEncoder

# 📦 IMPORT: standard Python libraries
import os        # for file/folder operations
import shutil    # for deleting folders (to reset the database)


class Retriever:
    """
    Manages document storage in ChromaDB and retrieval with re-ranking.

    📖 HOW CHROMADB WORKS:
    ChromaDB is a vector database that:
    1. Takes text + its embedding vector → stores them
    2. Takes a query embedding → finds the closest stored vectors
    3. Returns the original text for those closest matches

    Think of it like a smart filing cabinet:
    - Filing (indexing): put documents in folders based on their "meaning neighborhood"
    - Searching: put in a query, get back the most relevant documents

    📖 WHAT IS A COLLECTION?
    In ChromaDB, a "collection" is like a table in a regular database.
    Each collection has a name and stores related documents.
    We create one collection per chunking strategy so we can compare them.
    """

    # The cross-encoder model for re-ranking
    # ms-marco = a Microsoft dataset for question-answering
    # MiniLM-L-6-v2 = small, fast, good quality
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, embedder, collection_name="rag_collection", use_reranker=True):
        """
        Sets up the ChromaDB client and re-ranker.

        Parameters:
            embedder: our Embedder object from embedder.py
            collection_name: name for this database collection
            use_reranker: whether to use cross-encoder re-ranking
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.use_reranker = use_reranker

        # ── CHROMADB SETUP ──────────────────────────────────────
        # PersistentClient stores data on DISK (in a folder)
        # So even if you close Python, the data is still there.
        # Alternative: chromadb.Client() stores only in RAM (lost on close)
        #
        self.db_path = "./chroma_db_v2"
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Fallback to in-memory client if disk persistence encounters environment lock or migration mismatch
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # ── RE-RANKER SETUP ─────────────────────────────────────
        if self.use_reranker:
            print("⏳ Loading re-ranker model (downloads ~70MB first time)...")
            self.reranker = CrossEncoder(self.RERANKER_MODEL)
            print("✅ Re-ranker loaded.")
        else:
            self.reranker = None

    def index_chunks(self, chunks, metadata=None):
        """
        Stores a list of text chunks in ChromaDB.
        Called ONCE per document to build the index.

        Parameters:
            chunks:   list of strings — the text chunks to store
            metadata: optional extra info to store alongside each chunk
                      e.g., {"source": "ai_fundamentals.txt", "strategy": "semantic"}

        📖 WHAT IS metadata?
        Extra information stored with each chunk.
        Like a book's card in a card catalog — it has the text AND
        extra info about where it came from. We can filter by metadata later.
        """
        if not chunks:
            print("⚠️  No chunks to index.")
            return

        print(f"📥 Indexing {len(chunks)} chunks into ChromaDB...")

        # Generate embeddings for ALL chunks at once (fast batch processing)
        embeddings = self.embedder.embed_batch(chunks)
        # embeddings.shape = (num_chunks, 384)
        # embeddings[0] = embedding for chunks[0]
        # embeddings[1] = embedding for chunks[1], etc.

        # Generate unique IDs for each chunk based on collection count & metadata
        existing_count = self.collection.count()
        src_tag = metadata.get("source", "doc").replace(" ", "_") if metadata else "doc"
        ids = [f"chunk_{existing_count + i}_{src_tag}_{i}" for i in range(len(chunks))]
        # 📖 LIST COMPREHENSION again:
        # [f"chunk_{i}" for i in range(len(chunks))]
        # = ["chunk_0", "chunk_1", "chunk_2", ..., "chunk_N"]

        # Build metadata list — one dict per chunk
        if metadata is None:
            # If no metadata provided, create a simple dict for each chunk
            # containing just the chunk's index
            metadatas = [{"chunk_index": i} for i in range(len(chunks))]
        else:
            # Add chunk_index to the provided metadata for every chunk
            metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]
            # 📖 {**metadata, "chunk_index": i}
            # The ** "unpacks" the metadata dict and merges it with {"chunk_index": i}
            # Example: metadata = {"source": "file.txt"}
            # Result:  {"source": "file.txt", "chunk_index": 3}

        # Add everything to ChromaDB in one call
        # ChromaDB stores: text + embedding + id + metadata together
        self.collection.add(
            documents=chunks,          # the actual text
            embeddings=embeddings.tolist(),  # .tolist() converts numpy array → Python list
            ids=ids,                   # unique identifiers
            metadatas=metadatas        # extra info
        )

        print(f"✅ Indexed {len(chunks)} chunks. Collection size: {self.collection.count()}")

    def retrieve(self, query, top_k=5):
        """
        Finds the most relevant chunks for a given query.
        Returns top_k chunks, re-ranked if re-ranker is enabled.

        Parameters:
            query: the user's question as a string
            top_k: how many chunks to return

        Returns:
            A list of dicts, each containing:
            {
                "text": "the chunk text...",
                "score": 0.87,    ← relevance score (higher = better)
                "id": "chunk_5"
            }

        📖 HOW THE RETRIEVAL PIPELINE WORKS:
        1. Embed the query → get a 384-dim vector
        2. ChromaDB finds the N closest chunks (by cosine similarity)
        3. Re-ranker re-scores those N chunks with a more precise model
        4. Sort by re-ranker score and return top_k

        We ask ChromaDB for more chunks than we need (top_k * 3)
        so the re-ranker has more options to pick the best ones from.
        """
        # Check if collection has any documents
        if self.collection.count() == 0:
            print("⚠️  Collection is empty. Please index documents first.")
            return []

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_text(query)
        # query_embedding.shape = (384,) — a single vector

        # Step 2: Retrieve from ChromaDB
        # We fetch more than top_k to give the re-ranker options
        n_candidates = min(top_k * 3, self.collection.count())
        # min() ensures we don't ask for more chunks than exist

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            # [query_embedding.tolist()] — wrapped in a list because ChromaDB
            # supports querying multiple embeddings at once. We query one at a time.
            n_results=n_candidates,
            include=["documents", "distances", "metadatas"]
            # include: what data to return alongside the chunk text
            # "documents" = the text, "distances" = similarity scores
        )

        # results is a nested dict returned by ChromaDB:
        # results["documents"][0] = list of chunk texts (the [0] because we sent one query)
        # results["distances"][0] = list of distance scores
        # results["ids"][0]       = list of chunk IDs
        candidate_texts = results["documents"][0]
        candidate_ids = results["ids"][0]
        candidate_metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(candidate_texts)

        # Step 3: Re-rank if enabled
        if self.use_reranker and self.reranker and len(candidate_texts) > 0:
            return self._rerank(query, candidate_texts, candidate_ids, candidate_metadatas, top_k)
        else:
            # No re-ranker: just return top_k results with ChromaDB's scores
            distances = results["distances"][0]
            return [
                {
                    "text": candidate_texts[i],
                    "score": round(1 - distances[i], 4),
                    "id": candidate_ids[i],
                    "metadata": candidate_metadatas[i],
                    "source": candidate_metadatas[i].get("source", "Unknown") if isinstance(candidate_metadatas[i], dict) else "Unknown"
                }
                for i in range(min(top_k, len(candidate_texts)))
            ]

    def _rerank(self, query, candidate_texts, candidate_ids, candidate_metadatas, top_k):
        """
        Uses a cross-encoder to re-score and reorder candidates.
        Ensures multi-document fair representation so all uploaded documents are included.
        """
        pairs = [[query, text] for text in candidate_texts]
        scores = self.reranker.predict(pairs)

        all_items = [
            {
                "text": text,
                "score": round(float(score), 4),
                "id": cid,
                "metadata": meta,
                "source": meta.get("source", "Unknown") if isinstance(meta, dict) else "Unknown"
            }
            for score, text, cid, meta in zip(scores, candidate_texts, candidate_ids, candidate_metadatas)
        ]

        all_items.sort(key=lambda x: x["score"], reverse=True)

        # Multi-document balanced selection: ensure chunks from EVERY uploaded document are included
        distinct_sources = list(dict.fromkeys([item["source"] for item in all_items if item["source"] != "Unknown"]))

        if len(distinct_sources) > 1:
            source_groups = {src: [item for item in all_items if item["source"] == src] for src in distinct_sources}
            selected = []
            per_source_count = max(2, top_k // len(distinct_sources))

            for src in distinct_sources:
                selected.extend(source_groups[src][:per_source_count])

            selected_ids = {id(item) for item in selected}
            for item in all_items:
                if len(selected) >= top_k:
                    break
                if id(item) not in selected_ids:
                    selected.append(item)
                    selected_ids.add(id(item))

            selected.sort(key=lambda x: x["score"], reverse=True)
            return selected[:top_k]

        return all_items[:top_k]

    def clear(self):
        """
        Deletes all data in this collection (for re-indexing).
        Useful when switching chunking strategies.
        """
        # Delete and recreate the collection (fastest way to clear it)
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"🗑️  Collection '{self.collection_name}' cleared.")
        except Exception as e:
            print(f"⚠️  Could not clear collection: {e}")
            # Exception = an error that occurred during execution
            # try/except = "try this code, if an error happens, run the except block"

    def count(self):
        """Returns how many chunks are currently stored."""
        return self.collection.count()
