# ============================================================
# FILE: src/embedder.py
# PURPOSE: Converts text into numbers (vectors/embeddings)
#          that capture the MEANING of the text.
#
# WHY DO WE NEED THIS?
#   Computers can't understand words like "cat" or "king".
#   They only understand numbers.
#   An embedding model converts text → a list of numbers
#   in a way that preserves semantic meaning.
#   Words/sentences with similar meanings get similar numbers.
#
# WHAT MODEL DO WE USE?
#   "all-MiniLM-L6-v2" from sentence-transformers.
#   - Produces 384-dimensional embeddings (a list of 384 numbers)
#   - Fast and lightweight
#   - Downloads automatically on first use (~90MB)
#   - Runs entirely on your CPU — no GPU needed
# ============================================================


import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 📦 IMPORT: SentenceTransformer
from sentence_transformers import SentenceTransformer

# 📦 IMPORT: numpy
import numpy as np


class Embedder:
    """
    A wrapper around the SentenceTransformer embedding model.

    📖 WHY WRAP IT IN A CLASS?
    The SentenceTransformer class has many features we don't need.
    By wrapping it, we create a simpler, cleaner interface
    with only the methods our RAG system needs.
    This is called the "Facade" design pattern.

    📖 WHAT DOES IT MEAN TO "LOAD" A MODEL?
    A model is a neural network with millions of learned parameters
    (numbers). "Loading" means downloading those parameters from
    the internet and storing them in memory (RAM) so we can use them.
    First load = download (~90MB). After that, it loads from cache.
    """

    # CLASS-LEVEL CONSTANT: the model we're using
    # Written in UPPER_CASE by convention to signal it's a constant
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        """
        Loads the embedding model into memory.
        This happens once when you create an Embedder() object.
        """
        print(f"⏳ Loading embedding model: {self.MODEL_NAME}")
        print("   (First time: downloads ~90MB. After that: instant load)")

        # This single line downloads (if needed) and loads the model
        # SentenceTransformer returns a model object we can call .encode() on
        self.model = SentenceTransformer(self.MODEL_NAME)

        print(f"✅ Embedding model loaded. Embedding dimension: {self.get_dimension()}")

    def embed_text(self, text):
        """
        Converts a SINGLE string into a vector (list of numbers).

        Example:
            embedder = Embedder()
            vec = embedder.embed_text("What is machine learning?")
            # vec is now a numpy array of shape (384,)
            # i.e., 384 floating-point numbers

        📖 WHAT IS A numpy ARRAY?
        Similar to a Python list, but:
        - Can ONLY hold numbers (not mixed types)
        - Much faster for mathematical operations
        - Has a fixed "shape" (like dimensions)
        A 1D array of 384 numbers: [0.12, -0.45, 0.78, ..., 0.31]
        """
        # convert_to_numpy=True → return numpy array instead of PyTorch tensor
        # normalize_embeddings=True → scales the vector so its magnitude = 1
        #   (this makes cosine similarity calculation more accurate)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def embed_batch(self, texts):
        """
        Converts a LIST of strings into a 2D array of vectors.
        Much faster than calling embed_text() one at a time.

        Example:
            texts = ["Hello world", "AI is amazing", "Python is fun"]
            embeddings = embedder.embed_batch(texts)
            # embeddings.shape = (3, 384)
            # 3 texts, each with 384-dimensional embedding

        📖 WHAT IS A 2D ARRAY?
        Think of it like a spreadsheet:
        - Each ROW = one text's embedding
        - Each COLUMN = one dimension
        Shape (3, 384) = 3 rows, 384 columns
        """
        # show_progress_bar=True → shows a progress bar in terminal for large batches
        # batch_size=32 → processes 32 texts at a time (memory-efficient)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,  # only show bar for big batches
            batch_size=32
        )
        return embeddings

    def get_dimension(self):
        """
        Returns the number of dimensions in each embedding vector.
        For all-MiniLM-L6-v2, this is 384.

        We need to know this when setting up ChromaDB,
        because the database needs to know the size of vectors it'll store.
        """
        # Get the shape of a dummy embedding and return its size
        # np.zeros(1) creates an array [0.] just to have something to encode
        dummy = self.model.encode(["test"], convert_to_numpy=True)
        # dummy.shape = (1, 384) → dummy.shape[1] = 384
        return dummy.shape[1]

    def cosine_similarity(self, vec_a, vec_b):
        """
        Measures how similar two embedding vectors are.
        Returns a value between 0 and 1:
          1.0 = identical meaning
          0.5 = somewhat related
          0.0 = completely unrelated

        📖 WHY USE THIS METHOD HERE?
        Since our embeddings are already normalized (magnitude=1),
        cosine similarity simplifies to just the dot product!
        np.dot(a, b) when both are unit vectors = cos(angle between them)
        """
        # np.dot = dot product of two arrays
        # When both vectors are normalized, this equals cosine similarity
        return float(np.dot(vec_a, vec_b))
