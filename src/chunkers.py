# ============================================================
# FILE: src/chunkers.py
# PURPOSE: Splits a long document into smaller "chunks"
#
# WHY DO WE CHUNK?
#   Imagine trying to find a specific fact in a 300-page book.
#   You wouldn't give the entire book to Gemini and say "answer this."
#   Instead, we cut the book into small pieces, find the MOST
#   RELEVANT pieces for each question, and give ONLY those to Gemini.
#
#   This project compares 3 different ways to cut the document.
#   We measure which strategy finds the right piece most often.
#
# THE 3 STRATEGIES:
#   1. FixedSizeChunker  — cuts every N characters (dumb but fast)
#   2. SentenceChunker   — cuts at sentence boundaries (smarter)
#   3. SemanticChunker   — groups sentences by meaning (smartest)
# ============================================================


# 📦 IMPORT: re = "regular expressions"
# A powerful tool for finding and splitting patterns in text.
# We use it to find sentence endings like ".  " or "!  "
import re

# 📦 IMPORT: numpy
# A math library. We use it here to compute the average
# similarity between sentence embeddings.
import numpy as np


# ================================================================
# STRATEGY 1: FIXED SIZE CHUNKER
# ================================================================

class FixedSizeChunker:
    """
    Splits text into chunks of exactly N characters, with overlap.

    📖 WHAT IS A CLASS?
    A class is a blueprint for creating objects.
    Think of a class like a cookie cutter (the template),
    and the object like the actual cookie (a specific instance).

    This class is our blueprint for a fixed-size chunking tool.
    We can create one like this:
        chunker = FixedSizeChunker(chunk_size=500)
        chunks = chunker.split(my_text)

    📖 WHAT IS OVERLAP?
    Imagine the text is: "...The cat sat. The cat ate..."
    If we cut at exactly 500 chars, "The cat" might get split
    across two chunks, losing context. Overlap means we repeat
    the last N characters in the next chunk, so context isn't lost.
    """

    def __init__(self, chunk_size=500, overlap=50):
        # 📖 __init__ = "initialize"
        # This method runs AUTOMATICALLY the moment you create an object.
        # It's like the "setup" instructions for the object.
        #
        # self = refers to THIS specific object (the cookie, not the cutter)
        # self.chunk_size = stores the chunk_size value inside this object
        self.chunk_size = chunk_size   # how many characters per chunk
        self.overlap = overlap          # how many chars to repeat at boundaries

    def split(self, text):
        """
        Takes a long string and returns a list of smaller strings.

        📖 WHAT IS A LIST?
        A list is an ordered collection of items.
        Example: ["chunk one text", "chunk two text", "chunk three text"]
        Each item is a string (text). We return this list.

        📖 WHAT IS A WHILE LOOP?
        while <condition>:
            do something
        It keeps running AS LONG AS the condition is True.
        We use it to keep slicing chunks until we reach the end of the text.
        """
        chunks = []      # start with an empty list — we'll fill it up
        start = 0        # index of where the current chunk begins

        # Keep looping until 'start' has moved past the end of the text
        while start < len(text):
            # len(text) = total number of characters in the text
            # start + chunk_size = where this chunk should end
            end = start + self.chunk_size

            # text[start:end] = Python "slicing"
            # It extracts characters from index 'start' up to (but not including) 'end'
            # Example: "hello"[1:3] = "el"
            chunk = text[start:end]

            # .strip() removes whitespace (spaces, newlines) from both ends
            # We only add chunks that have actual content (not empty strings)
            if chunk.strip():
                chunks.append(chunk.strip())
                # .append() adds an item to the END of a list

            # Move 'start' forward by chunk_size MINUS overlap
            # This creates the overlap between consecutive chunks
            start += self.chunk_size - self.overlap

        return chunks  # return the complete list of chunks


# ================================================================
# STRATEGY 2: SENTENCE CHUNKER
# ================================================================

class SentenceChunker:
    """
    Splits text at sentence boundaries, then groups sentences
    into chunks of approximately N sentences each.

    WHY IS THIS BETTER THAN FIXED-SIZE?
    Fixed-size chunking cuts in the middle of sentences.
    Example: "The Turing Test, invented by Alan Tu" ← cut here
    The next chunk starts with: "ring, tests whether..."
    
    Sentence chunking always ends at a complete sentence,
    preserving meaning and making retrieval more accurate.
    """

    def __init__(self, sentences_per_chunk=5, overlap_sentences=1):
        # sentences_per_chunk: how many sentences per chunk
        # overlap_sentences: how many sentences to repeat between chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def _split_into_sentences(self, text):
        """
        Breaks a long text into individual sentences.

        📖 WHAT IS A "PRIVATE" METHOD (with underscore _)?
        The underscore prefix is a Python convention meaning:
        "This method is for internal use only — don't call it from outside."
        It's still accessible, just a signal to other programmers.

        📖 HOW DOES re.split() WORK?
        re.split(pattern, text) splits text wherever the pattern matches.
        
        Our pattern:  r'(?<=[.!?])\s+'
        Breaking it down:
          (?<=[.!?]) = "look BEHIND and check if there's a . or ! or ?"
                       This is called a "lookbehind assertion"
          \s+        = "one or more whitespace characters (spaces, newlines)"
        
        So: split AFTER a sentence-ending punctuation followed by space.
        "Hello world. How are you? I am fine." 
        → ["Hello world.", "How are you?", "I am fine."]
        """
        # re.split splits text using the pattern as delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out empty strings and very short fragments (less than 10 chars)
        # A fragment like "\n" or "   " isn't a real sentence
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        #
        # 📖 LIST COMPREHENSION: [s.strip() for s in sentences if len(s.strip()) > 10]
        # This is a compact way to write a for loop that builds a list.
        # It means: "for each sentence s in sentences, include s.strip()
        #            BUT ONLY IF its length is more than 10 characters"
        # Equivalent to:
        #   result = []
        #   for s in sentences:
        #       if len(s.strip()) > 10:
        #           result.append(s.strip())

        return sentences

    def split(self, text):
        """Groups sentences into chunks."""
        sentences = self._split_into_sentences(text)
        chunks = []
        i = 0  # index into the sentences list

        while i < len(sentences):
            # Grab the next N sentences (or fewer if near the end)
            # sentences[i : i + self.sentences_per_chunk]
            # = a "slice" of the list from index i to i+N
            sentence_group = sentences[i : i + self.sentences_per_chunk]

            # " ".join([...]) = joins a list of strings with a space between each
            # Example: " ".join(["Hello.", "World."]) = "Hello. World."
            chunk = " ".join(sentence_group)

            if chunk.strip():
                chunks.append(chunk.strip())

            # Move forward by sentences_per_chunk MINUS overlap
            # This creates overlap between consecutive chunks
            i += self.sentences_per_chunk - self.overlap_sentences

        return chunks


# ================================================================
# STRATEGY 3: SEMANTIC CHUNKER
# ================================================================

class SemanticChunker:
    """
    Groups sentences by MEANING similarity into chunks.
    
    This is the most sophisticated strategy. Instead of cutting
    at fixed sizes or sentence counts, it finds natural "topic
    boundaries" in the text — places where the subject shifts.

    HOW IT WORKS:
    1. Split text into sentences
    2. Embed each sentence (convert to numbers that represent meaning)
    3. Compute similarity between adjacent sentences
    4. When similarity DROPS sharply → that's a topic boundary → start new chunk
    5. Merge small chunks to avoid tiny fragments

    WHY IS THIS THE BEST?
    If text goes from talking about "AlexNet" to "BERT transformers",
    the similarity between those adjacent sentences will DROP.
    We detect that drop and create a chunk boundary there.
    The result: each chunk is semantically coherent — about ONE topic.

    📖 WHAT IS AN EMBEDDING MODEL PARAMETER?
    We pass the embedding model IN from outside. This is called
    "dependency injection" — the chunker doesn't create its own model,
    it uses whatever model we give it. This makes it flexible.
    """

    def __init__(self, embedding_model, threshold=0.4, min_chunk_size=100):
        # embedding_model: an object that can convert text → numbers
        #                  we pass this in from embedder.py
        # threshold: similarity score below which we cut a new chunk
        #            0.4 means: "if sentences are less than 40% similar, split"
        # min_chunk_size: don't create chunks smaller than this many characters
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

        # We reuse the sentence-splitter from SentenceChunker
        # _sentence_chunker is a helper we create internally
        self._sentence_chunker = SentenceChunker(sentences_per_chunk=1)

    def split(self, text):
        """
        Splits text at semantic (meaning) boundaries.
        """
        # Step 1: Get individual sentences
        sentences = self._sentence_chunker._split_into_sentences(text)

        # If there are too few sentences, just return the whole text as one chunk
        if len(sentences) <= 2:
            return [text.strip()]

        # Step 2: Embed ALL sentences at once
        # self.embedding_model.encode() converts a list of strings
        # into a 2D numpy array: shape = (num_sentences, embedding_dim)
        # e.g., for 50 sentences with 384-dim embeddings: shape = (50, 384)
        embeddings = self.embedding_model.encode(sentences)

        # Step 3: Compute similarity between ADJACENT sentences
        # We compare sentence[0] with sentence[1], sentence[1] with sentence[2], etc.
        similarities = []
        for i in range(len(embeddings) - 1):
            # Cosine similarity = how "similar" two vectors are
            # Result is between -1 (opposite) and 1 (identical)
            # We compute it manually here using the dot product formula
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Step 4: Find chunk boundaries (where similarity drops below threshold)
        chunks = []
        current_chunk_sentences = [sentences[0]]  # start with first sentence

        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                # Similarity dropped → topic shift → end current chunk
                chunk_text = " ".join(current_chunk_sentences)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                elif chunks:
                    # Chunk is too small → merge with previous chunk
                    chunks[-1] += " " + chunk_text
                current_chunk_sentences = []  # reset for new chunk

            # Always add the NEXT sentence (i+1) to the current group
            current_chunk_sentences.append(sentences[i + 1])

        # Don't forget the last group of sentences after the loop ends
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if chunks and len(chunk_text) < self.min_chunk_size:
                chunks[-1] += " " + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks

    def _cosine_similarity(self, vec_a, vec_b):
        """
        Measures how similar two vectors are using cosine similarity.

        📖 WHAT IS COSINE SIMILARITY?
        Imagine two arrows pointing in space. If they point in the
        same direction → similarity = 1.0 (identical meaning).
        If they're perpendicular → similarity = 0.0 (no relation).
        If opposite → similarity = -1.0 (opposite meaning).

        The formula: dot_product(A, B) / (magnitude(A) * magnitude(B))
        
        np.dot(a, b) = dot product (sum of element-wise multiplications)
        np.linalg.norm(a) = magnitude (length) of vector a
        """
        dot_product = np.dot(vec_a, vec_b)
        magnitude_a = np.linalg.norm(vec_a)
        magnitude_b = np.linalg.norm(vec_b)

        # Avoid division by zero (if a vector is all zeros)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)


# ================================================================
# STRATEGY 4: LANGCHAIN RECURSIVE CHUNKER
# ================================================================

class LangChainRecursiveChunker:
    """
    Splits text using LangChain's industry-standard RecursiveCharacterTextSplitter.
    Recursively tries splitting by paragraph ("\n\n"), newline ("\n"), space (" "),
    and characters ("") to maximize semantic continuity within chunk budgets.
    """

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        except Exception:
            self.splitter = None

    def split(self, text):
        if not text or not text.strip():
            return []
        if self.splitter:
            return self.splitter.split_text(text)
        # Fallback to fixed size if LangChain splitter is unavailable
        fallback = FixedSizeChunker(self.chunk_size, self.overlap)
        return fallback.split(text)


# ================================================================
# CHUNKER FACTORY — a helper to get the right chunker by name
# ================================================================

def get_chunker(strategy_name, embedding_model=None):
    """
    Returns the appropriate chunker object based on a strategy name.

    📖 WHAT IS A FACTORY FUNCTION?
    A factory function creates and returns objects based on input.
    Instead of writing if/else everywhere in your code, you call
    get_chunker("semantic") and get back the right object.
    """
    # Maps strategy name strings to chunker class instances
    chunkers = {
        "fixed":     FixedSizeChunker(chunk_size=500, overlap=50),
        "sentence":  SentenceChunker(sentences_per_chunk=5, overlap_sentences=1),
        "semantic":  SemanticChunker(embedding_model=embedding_model, threshold=0.4),
        "langchain": LangChainRecursiveChunker(chunk_size=500, overlap=50),
        "recursive": LangChainRecursiveChunker(chunk_size=500, overlap=50)
    }

    chunker = chunkers.get(strategy_name.lower())

    if chunker is None:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {list(chunkers.keys())}"
        )

    return chunker

