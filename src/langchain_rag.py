# ============================================================
# FILE: src/langchain_rag.py
# PURPOSE: Enterprise RAG pipeline powered by LangChain & LCEL
#
# ARCHITECTURE (LangChain Expression Language - LCEL):
#   Document ──► LangChain RecursiveCharacterTextSplitter ──► Documents
#        │
#        ▼
#   HuggingFace / Sentence-Transformers Embeddings ──► ChromaDB Vector Store
#        │
#        ▼
#   2-Stage Retrieval (Chroma Similarity Top-K + Cross-Encoder Neural Re-Ranking)
#        │
#        ▼
#   Context Aggregation & PromptTemplate Formatting
#        │
#        ▼
#   LLM (Google Gemini 3.8 Flash) ──► StrOutputParser ──► Verified Answer
# ============================================================

import os
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Graceful fallback if packages are currently finalizing installation
    Document = Any
    PromptTemplate = Any
    StrOutputParser = Any
    RunnablePassthrough = Any
    RecursiveCharacterTextSplitter = Any

# Internal modules
try:
    from src.embedder import Embedder
    from src.retriever import Retriever
except ImportError:
    from embedder import Embedder
    from retriever import Retriever


class LangChainRAGPipeline:
    """
    Enterprise LangChain LCEL RAG pipeline with 2-stage retrieval
    and Cross-Encoder neural re-ranking.
    """

    DEFAULT_PROMPT_TEMPLATE = """You are RAGDoc AI, an expert analytical intelligence copilot.
Answer the user's question with authoritative accuracy, drawing strictly from the retrieved context below.

---------------------
RETRIEVED CONTEXT:
{context}
---------------------

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a direct, structured, and comprehensive answer based ONLY on the context.
2. If the context contains specific metrics, dates, or figures, cite them verbatim.
3. If the context does not contain sufficient information, state: "The provided documents do not contain sufficient evidence to answer this question."
4. Do not speculate or introduce ungrounded external assumptions.

ACCURATE ANSWER:"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_reranker: bool = True,
        collection_name: str = "langchain_rag_collection"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker
        self.collection_name = collection_name

        # 1. Text splitter
        try:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        except Exception:
            self.splitter = None

        # 2. Embedding & ChromaDB retriever engine
        self.embedder = Embedder()
        self.retriever = Retriever(
            embedder=self.embedder,
            collection_name=collection_name,
            use_reranker=use_reranker
        )

        # 3. LangChain Prompt
        try:
            self.prompt = PromptTemplate.from_template(self.DEFAULT_PROMPT_TEMPLATE)
            self.parser = StrOutputParser()
        except Exception:
            self.prompt = None
            self.parser = None

        self.indexed_doc_name: Optional[str] = None
        self.chunks_count: int = 0

    def index_document(self, text: str, source_name: str = "document") -> Dict[str, Any]:
        """
        Chunks text using LangChain RecursiveCharacterTextSplitter
        and indexes chunks into the ChromaDB vector database.
        """
        t0 = time.time()

        if self.splitter:
            raw_chunks = self.splitter.split_text(text)
        else:
            # Fallback simple windowing
            raw_chunks = [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        # Convert to LangChain Document structures with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": source_name, "chunk_id": idx, "char_count": len(chunk)}
            )
            for idx, chunk in enumerate(raw_chunks)
        ]

        # Index into ChromaDB
        self.retriever.index_chunks(raw_chunks, metadata={"source": source_name})
        self.indexed_doc_name = source_name
        self.chunks_count = len(documents)

        index_latency = round(time.time() - t0, 3)
        return {
            "chunks_count": len(documents),
            "latency_seconds": index_latency,
            "sample_chunk": raw_chunks[0] if raw_chunks else ""
        }

    def format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean context block."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", "")
            score = doc.get("rerank_score", doc.get("similarity_score", 0.0))
            formatted.append(f"[Source Section {i} | Relevance: {score:.3f}]\n{text}")
        return "\n\n".join(formatted)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the LangChain RAG pipeline:
        1. Retrieve top-K candidates from ChromaDB
        2. Neural Cross-Encoder Re-Ranking (2-stage)
        3. Format into LangChain LCEL PromptTemplate
        4. Generate synthesis via LLM
        """
        t0 = time.time()

        # Step 1 & 2: 2-stage retrieval
        retrieved_results = self.retriever.retrieve(
            query=question,
            top_k=top_k
        )

        formatted_context = self.format_docs(retrieved_results)

        # Step 3: Prompt Formatting
        prompt_text = (
            self.prompt.format(context=formatted_context, question=question)
            if self.prompt
            else self.DEFAULT_PROMPT_TEMPLATE.replace("{context}", formatted_context).replace("{question}", question)
        )

        # Step 4: Generation via Google Gemini 3.8 Flash
        answer_text = ""
        llm_engine_used = "Google Gemini 3.8 Flash (Cloud)"

        api_key_to_use = gemini_api_key or os.getenv("GEMINI_API_KEY")

        if api_key_to_use:
            import requests
            models_to_try = [
                "gemini-3.8-flash",
                "gemini-3.7-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite"
            ]
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "maxOutputTokens": 8192,
                    "temperature": 0.2
                }
            }
            for model_name in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key_to_use}"
                    res = requests.post(url, json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text_chunks = [p.get("text", "") for p in parts if "text" in p]
                            full_text = "".join(text_chunks).strip()
                            if full_text:
                                answer_text = full_text
                                llm_engine_used = f"Google Gemini 3.8 Flash ({model_name} Cloud)"
                                break
                except Exception:
                    continue

            if not answer_text:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key_to_use)
                    model = genai.GenerativeModel(
                        "gemini-3.8-flash",
                        generation_config={"max_output_tokens": 8192, "temperature": 0.2}
                    )
                    response = model.generate_content(prompt_text)
                    if response and response.text:
                        answer_text = response.text.strip()
                        llm_engine_used = "Google Gemini 3.8 Flash (Cloud)"
                except Exception as e:
                    answer_text = f"Gemini generation error: {e}"
        else:
            answer_text = "⚠️ GEMINI_API_KEY is not configured in .env file."

        total_latency = round(time.time() - t0, 3)

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_results,
            "context": formatted_context,
            "prompt": prompt_text,
            "llm_engine": llm_engine_used,
            "latency_seconds": total_latency,
            "top_score": retrieved_results[0].get("rerank_score", retrieved_results[0].get("similarity_score", 0.0)) if retrieved_results else 0.0
        }
