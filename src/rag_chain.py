# ============================================================
# FILE: src/rag_chain.py
# PURPOSE: The central RAG pipeline — orchestrates ingestion,
#          LangChain chunking, neural 2-stage retrieval, and Gemini 3.8 Flash.
#
# ARCHITECTURE:
#   Document ──► LangChain Recursive / Semantic Chunker ──► Embeddings
#        │
#        ▼
#   ChromaDB Vector Store (Stage 1: Top-15 Candidate Retrieval)
#        │
#        ▼
#   Cross-Encoder Neural Re-Ranker (Stage 2: Top-5 Precision Filtering)
#        │
#        ▼
#   Google Gemini 3.8 Flash Cloud LLM ──► Source-Grounded Answer
# ============================================================


# 📦 Standard library
import os
import sys

# Fix Windows emoji encoding issue
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure project root is in Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 📦 Our modules
try:
    from src.chunkers import get_chunker
    from src.embedder import Embedder
    from src.retriever import Retriever
except ModuleNotFoundError:
    from chunkers import get_chunker
    from embedder import Embedder
    from retriever import Retriever


class RAGChain:
    """
    The complete RAG pipeline from document to answer.

    📖 THE FULL PIPELINE:
    ┌────────────────────────────────────────────────────────┐
    │  DOCUMENTS (.pdf, .docx, .txt)                         │
    │      │                                                 │
    │      ▼                                                 │
    │  [LANGCHAIN CHUNKER] ──splits into──► chunks list      │
    │      │                                                 │
    │      ▼                                                 │
    │  [EMBEDDER] ──converts──► embedding vectors            │
    │      │                                                 │
    │      ▼                                                 │
    │  [RETRIEVER/ChromaDB] ──stores vectors──► indexed DB   │
    │                                                        │
    │  USER QUESTION                                         │
    │      │                                                 │
    │      ▼                                                 │
    │  [STAGE 1: Dense Retrieval] ──top 15 candidates        │
    │      │                                                 │
    │      ▼                                                 │
    │  [STAGE 2: Cross-Encoder] ──re-ranks──► top 5 chunks   │
    │      │                                                 │
    │      ▼                                                 │
    │  [GEMINI 3.8 FLASH] ──reads chunks──► ACCURATE ANSWER  │
    └────────────────────────────────────────────────────────┘
    """

    # Google Gemini 3.8 Flash model for ultra-fast, high-precision answer generation
    GEMINI_MODEL = "gemini-3.8-flash"

    def __init__(self, chunking_strategy="semantic", use_reranker=True):
        """
        Sets up the complete RAG pipeline powered by Google Gemini 3.8 Flash.

        Parameters:
            chunking_strategy:  "fixed", "sentence", "semantic", or "langchain"
            use_reranker:       whether to use cross-encoder re-ranking
        """
        self.chunking_strategy = chunking_strategy
        print(f"[Gemini] Engine active: Google {self.GEMINI_MODEL}")

        # ── INITIALIZE EMBEDDER ───────────────────────────────
        # Create ONE embedder — shared across chunker and retriever
        # so we don't load the model twice (it's ~90MB)
        print(f"\n🔧 Initializing RAG pipeline (strategy: {chunking_strategy})")
        self.embedder = Embedder()

        # ── INITIALIZE CHUNKER ────────────────────────────────
        # get_chunker() returns the right chunker object for the strategy
        # For semantic chunking, we pass the embedder (needed to compute similarities)
        self.chunker = get_chunker(
            strategy_name=chunking_strategy,
            embedding_model=self.embedder.model  # pass the raw model for SemanticChunker
        )

        # ── INITIALIZE RETRIEVER ─────────────────────────────
        # Each strategy gets its own collection name in ChromaDB
        # so strategies don't interfere with each other
        collection_name = f"rag_{chunking_strategy}"
        self.retriever = Retriever(
            embedder=self.embedder,
            collection_name=collection_name,
            use_reranker=use_reranker
        )

        print(f"✅ Pipeline ready. Strategy: {chunking_strategy}\n")

    def load_document(self, file_path, original_filename=None):
        """
        Reads, chunks, and indexes one or more documents into ChromaDB.
        Accepts either a single file path string or a list of file inputs.
        """
        if isinstance(file_path, (list, tuple)):
            return self.load_documents(file_path)

        doc_name = original_filename if original_filename else os.path.basename(file_path)
        print(f"📂 Loading document: {doc_name}")

        # Determine file type from its extension
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".txt":
            text = self._read_txt(file_path)
        elif file_extension == ".pdf":
            text = self._read_pdf(file_path)
        elif file_extension in [".docx", ".doc"]:
            text = self._read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Use .txt, .pdf, or .docx")

        print(f"✅ Document loaded. Total characters: {len(text):,}")

        # Chunk the text using the selected strategy
        print(f"✂️  Chunking with strategy: {self.chunking_strategy}...")
        chunks = self.chunker.split(text)
        print(f"✅ Created {len(chunks)} chunks.")

        # Show statistics about chunk sizes
        if chunks:
            sizes = [len(c) for c in chunks]
            print(f"   Avg chunk size: {sum(sizes)//len(sizes)} chars")
            print(f"   Min: {min(sizes)} | Max: {max(sizes)} chars")

        # Index chunks into ChromaDB with original filename stored in metadata
        self.retriever.index_chunks(
            chunks=chunks,
            metadata={
                "source": doc_name,
                "strategy": self.chunking_strategy
            }
        )

        return chunks

    def load_documents(self, file_inputs):
        """
        Reads, chunks, and indexes multiple documents (up to 5 or more).
        """
        self.retriever.clear()
        if isinstance(file_inputs, str):
            file_inputs = [file_inputs]

        all_chunks = []
        print(f"\n📚 Processing {len(file_inputs)} document(s)...")
        for item in file_inputs:
            if isinstance(item, (list, tuple)):
                path, orig_name = item[0], item[1]
                chunks = self.load_document(path, original_filename=orig_name)
            else:
                chunks = self.load_document(item)
            all_chunks.extend(chunks)

        print(f"✅ Total chunks indexed across all documents: {len(all_chunks)}\n")
        return all_chunks

    def _read_txt(self, file_path):
        """
        Reads a plain text (.txt) file and returns its content as a string.

        📖 ENCODING: "utf-8"
        Files are stored as bytes (numbers). To convert bytes → text,
        you need to know the "encoding" — the rule for converting.
        UTF-8 is the most common encoding and supports all languages.
        Always specify it explicitly to avoid errors on different computers.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            # f.read() = reads the ENTIRE file content as one big string

    def _read_pdf(self, file_path):
        """
        Reads a PDF file using PyMuPDF (fitz) and extracts text.
        If a page contains scanned photos/images (e.g. ID cards, certificates)
        with little or no extractable vector text, automatically runs RapidOCR!
        """
        try:
            import pymupdf as fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

        text_parts = []
        ocr_engine = None

        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc):
                page_text = page.get_text()

                # If page text is empty or too short (<30 chars), run OCR on the rendered page image
                if len(page_text.strip()) < 30:
                    try:
                        if ocr_engine is None:
                            from rapidocr_onnxruntime import RapidOCR
                            ocr_engine = RapidOCR()

                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        ocr_result, _ = ocr_engine(img_bytes)

                        if ocr_result:
                            ocr_text = "\n".join([line[1] for line in ocr_result if line[1].strip()])
                            if ocr_text.strip():
                                print(f"📷 [OCR] Extracted {len(ocr_text)} chars from image/scanned page {page_number+1}")
                                page_text = ocr_text
                    except Exception as ocr_err:
                        print(f"⚠️ OCR warning on page {page_number+1}: {ocr_err}")

                if page_text.strip():
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def _read_docx(self, file_path):
        """
        Reads a Word (.docx) file using python-docx and extracts text.
        """
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        doc = docx.Document(file_path)
        text_parts = []

        # Read paragraph text
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())

        # Read table cell text if tables exist
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))

        return "\n\n".join(text_parts)

    def _generate_llm_response(self, prompt, api_key=None, provider=None):
        """
        Generates response using Google Gemini 3.8 Flash cloud inference.
        """
        import requests
        errors = []

        def get_secret(name):
            val = os.environ.get(name)
            if not val:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets") and name in st.secrets:
                        val = st.secrets[name]
                except Exception:
                    pass
            return val

        gemini_key = api_key or get_secret("GEMINI_API_KEY")

        if not gemini_key:
            return "⚠️ Error: GEMINI_API_KEY is not configured. Please ensure it is present in your .env file."

        # Prioritize Google Gemini 3.8 Flash (the latest model), followed by reliable fallbacks
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
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.2
            }
        }

        for model_name in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                res = requests.post(url, json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_chunks = [p.get("text", "") for p in parts if "text" in p]
                        full_text = "".join(text_chunks).strip()
                        if full_text:
                            return full_text
                else:
                    errors.append(f"Gemini {model_name} ({res.status_code}): {res.text[:80]}")
            except Exception as e:
                errors.append(f"Gemini {model_name}: {str(e)[:60]}")

        # Fallback to Google GenAI SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                "gemini-3.8-flash",
                generation_config={"max_output_tokens": 8192, "temperature": 0.2}
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            errors.append(f"GenAI SDK ({str(e)[:60]})")

        return f"⚠️ Gemini Cloud Generation Error: {' | '.join(errors)}"

    def answer(self, question, top_k=5):
        """
        The main method: takes a question, returns an answer.
        """
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=top_k)

        if not retrieved_chunks:
            return "I couldn't find relevant information in the document.", ""

        context_parts = [f"[Source {i+1}]\n{chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
        context = "\n\n".join(context_parts)

        prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question,
say: "I don't have enough information to answer this from the provided context."

Do NOT add information from outside the context. Be concise and direct.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        answer_text = self._generate_llm_response(prompt)
        return answer_text, context

    def format_answer_with_sources(self, question, top_k=6, chat_history=None, api_key=None, provider=None):
        """
        A user-friendly version of answer() that includes source chunks, multi-document synthesis, and follow-up chat history.
        """
        # Retrieve more chunks (top_k=8) to ensure all uploaded files are represented
        effective_k = max(top_k, 8)
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=effective_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the document(s).",
                "sources": [],
                "question": question
            }

        context_parts = []
        distinct_sources = set()
        for i, c in enumerate(retrieved_chunks):
            src_file = c.get("source") or c.get("metadata", {}).get("source", "Unknown Document")
            distinct_sources.add(src_file)
            context_parts.append(f"[Source {i+1} - File: {src_file}]\n{c['text']}")

        context = "\n\n".join(context_parts)
        sources_list_str = ", ".join(list(distinct_sources))

        history_text = ""
        if chat_history:
            formatted_turns = []
            for msg in chat_history[-6:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_turns.append(f"{role}: {msg.get('content', '')}")
            history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join(formatted_turns) + "\n"

        prompt = f"""You are a precise question-answering and document comparison assistant analyzing the user's uploaded document(s).
The provided context contains chunks from the following uploaded file(s): {sources_list_str}.

Instructions:
1. Answer the user's question using ONLY the provided document context and conversation history.
2. If multiple documents are present in the context, compare, contrast, and synthesize details from ALL provided documents thoroughly.
3. Be direct, structured, and cite specific document filenames when referring to information.
4. If asked to compare documents, analyze similarities and differences using the provided context chunks from each document.
5. Provide a complete, fully formed answer and never stop or cut off midway.

At the very end of your response, provide 2 to 3 logical follow-up questions the user might want to ask next based on the document content. Format them clearly under this heading:
💡 Suggested Follow-up Questions:
1. First follow-up question
2. Second follow-up question
3. Third follow-up question

DOCUMENT CONTEXT (Files: {sources_list_str}):
{context}
{history_text}
QUESTION: {question}

ANSWER:"""

        answer = self._generate_llm_response(prompt, api_key=api_key, provider=provider)

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "question": question,
            "context": context
        }


# ================================================================
# STANDALONE USAGE (run this file directly for a quick test)
# ================================================================

# if __name__ == "__main__":
#   This block ONLY runs when you execute this file directly:
#       python src/rag_chain.py
#   It does NOT run when this file is imported by another file.
#   This is a standard Python pattern for making files both
#   importable (as a module) AND runnable (as a script).

if __name__ == "__main__":
    print("RAGChain module loaded successfully.")
