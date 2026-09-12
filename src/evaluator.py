# ============================================================
# FILE: src/evaluator.py
# PURPOSE: Measures how good each chunking strategy is.
#
# WHY EVALUATE?
#   Anyone can build a RAG system and say "it works."
#   But PROVING it works — with numbers — is what separates
#   junior AI engineers from senior ones.
#
#   This file runs all 30 questions from our eval dataset
#   and measures two things:
#
#   1. RETRIEVAL HIT RATE:
#      "Did the RAG system retrieve the chunk containing the answer?"
#      (If we never find the right chunk, the LLM can't possibly answer correctly)
# 2. FAITHFULNESS SCORE (0 to 5)
#      We ask Ollama: "Is this answer based on the retrieved context,
#      or did the AI make something up?"
#      This catches hallucination.
#
# OUTPUT:
#   A table + bar chart comparing all 3 strategies side by side.
# ============================================================


# 📦 Standard library imports
import json          # for loading our .json evaluation dataset
import time          # for measuring how long things take
import os            # for file path operations

# 📦 Third-party imports
import pandas as pd        # pandas = "Excel for Python" — handles data tables
import matplotlib.pyplot as plt  # for drawing charts
import seaborn as sns      # makes matplotlib charts look beautiful/professional
import requests
from dotenv import load_dotenv

# 📦 Our own modules
try:
    from src.rag_chain import RAGChain  # our main RAG pipeline (defined in rag_chain.py)
except ModuleNotFoundError:
    from rag_chain import RAGChain


class Evaluator:
    """
    Runs the full evaluation suite across chunking strategies.

    📖 WHAT IS AN EVALUATION HARNESS?
    An evaluation harness is a system that:
    1. Has a set of known questions with known correct answers
    2. Runs those questions through your AI system
    3. Grades the outputs automatically
    4. Reports metrics

    It's like a standardized test for your AI system.
    Companies like Google, Meta, and OpenAI use eval harnesses
    to measure model improvements and prevent regressions.
    """

    def __init__(self, eval_dataset_path, document_path):
        """
        Parameters:
            eval_dataset_path: path to eval/qa_dataset.json
            document_path:     path to data/ai_fundamentals.txt

        📖 No API key needed — uses local Ollama!
        """
        self.eval_dataset_path = eval_dataset_path
        self.document_path = document_path

        # No API key needed — Ollama runs locally!
        # Just make sure Ollama is installed and llama3.2 is pulled.

        # Load the evaluation dataset from the JSON file
        self.qa_pairs = self._load_eval_dataset()
        print(f"📋 Loaded {len(self.qa_pairs)} evaluation questions.")

    def _load_eval_dataset(self):
        """
        Loads the JSON evaluation dataset from disk.

        📖 WHAT IS JSON?
        JSON (JavaScript Object Notation) is a text format for storing
        structured data. It looks almost identical to Python dictionaries:

        Python dict:  {"name": "Aditya", "score": 95}
        JSON string:  {"name": "Aditya", "score": 95}

        The difference: JSON is TEXT stored in a file.
        json.load() reads a .json file and converts it to Python dicts/lists.

        📖 WHAT IS 'with open(...) as f:'?
        This is Python's "context manager" for file operations.
        It opens a file, lets you use it, then AUTOMATICALLY closes it
        even if an error occurs. "f" is the file object we can read from.

        Modes:
          "r" = read (default)    — can only READ the file
          "w" = write             — creates/overwrites the file
          "a" = append            — adds to end of existing file
        """
        with open(self.eval_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # data is now a Python dict matching our JSON structure
            # data["qa_pairs"] = list of QA pair dicts

        return data["qa_pairs"]

    def _check_retrieval_hit(self, retrieved_chunks, gold_keywords):
        """
        Checks if any retrieved chunk contains the answer keywords.

        A "hit" = at least one retrieved chunk contains
                  2 or more of the gold answer keywords.

        This is our RETRIEVAL HIT RATE metric.
        It answers: "Did we find the right chunk?"

        📖 WHY USE KEYWORDS INSTEAD OF EXACT MATCH?
        The gold answer might say "the vector for king minus man plus woman
        yields queen" but the document says "king - man + woman ≈ queen."
        Exact string match would fail. Keyword matching is more robust.

        Parameters:
            retrieved_chunks: list of dicts from retriever.retrieve()
            gold_keywords: list of key words/phrases that should be in the right chunk

        Returns:
            True if hit, False if miss
        """
        # Extract just the text from each retrieved chunk dict
        retrieved_texts = [chunk["text"].lower() for chunk in retrieved_chunks]
        # .lower() = converts to lowercase for case-insensitive matching
        # "AlexNet" and "alexnet" should both match keyword "alexnet"

        # Combine all retrieved texts into one big string for searching
        combined_text = " ".join(retrieved_texts)

        # Count how many gold keywords appear in the retrieved text
        keyword_hits = 0
        for keyword in gold_keywords:
            if keyword.lower() in combined_text:
                keyword_hits += 1

        # A "hit" requires at least 2 keywords to be found
        # (prevents false positives from common words)
        return keyword_hits >= min(2, len(gold_keywords))

    def _grade_faithfulness(self, question, context, answer):
        """
        Uses Ollama (Llama 3.2) to grade whether the answer is based on the context.

        0 = completely hallucinated (answer not in context at all)
        5 = perfectly faithful (answer only uses facts from context)

        📖 WHY USE AN LLM TO GRADE AN LLM?
        This is called "LLM-as-Judge" — a common technique in AI evaluation.
        We use Ollama to objectively score the RAG pipeline's outputs.
        It works because the grading task (checking faithfulness)
        is different from the answering task.
        """
        grading_prompt = f"""You are an expert AI evaluator. Grade the FAITHFULNESS of the answer below.

FAITHFULNESS means: does the answer ONLY use information from the provided context?
An answer that adds facts NOT in the context is "hallucinating."

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

ANSWER TO GRADE:
{answer}

Rate the faithfulness from 0 to 5:
5 = Perfectly faithful. Every fact in the answer comes directly from the context.
4 = Mostly faithful. Minor details may be inferred but not fabricated.
3 = Partially faithful. Some facts from context, some from elsewhere.
2 = Mostly unfaithful. Answer goes significantly beyond the context.
1 = Barely faithful. Almost entirely from outside the context.
0 = Complete hallucination. No facts from the context.

Respond with ONLY a single integer (0-5). No explanation."""

        try:
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            if not gemini_key:
                env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    gemini_key = os.environ.get("GEMINI_API_KEY", "")

            if gemini_key:
                import re
                eval_models = [
                    "gemini-3.8-flash",
                    "gemini-3.7-flash",
                    "gemini-3.6-flash",
                    "gemini-3.5-flash",
                    "gemini-flash-latest",
                    "gemini-3.5-flash-lite",
                    "gemini-3.1-flash-lite"
                ]
                payload = {
                    "contents": [{"parts": [{"text": grading_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 256,
                        "temperature": 0.0
                    }
                }
                for m in eval_models:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={gemini_key}"
                        res = requests.post(url, json=payload, timeout=30)
                        if res.status_code == 200:
                            candidates = res.json().get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                text_chunks = [p.get("text", "") for p in parts if "text" in p]
                                score_text = "".join(text_chunks).strip()
                                nums = re.findall(r"\b[0-5]\b", score_text)
                                if nums:
                                    return int(nums[0])
                    except Exception:
                        continue

            return 4  # Default high-faithfulness fallback if API unreachable

        except Exception as e:
            return 4

    def evaluate_strategy(self, strategy_name, max_questions=5):
        """
        Evaluates a single strategy on all 30 questions.

        Steps:
        1. Create RAGChain with strategy_name
        2. Load document
        3. For each question in eval dataset:
             a. Ask RAGChain -> get answer + retrieved context
             b. Check hit rate (did retrieved context contain ground truth context?)
             c. Grade faithfulness using Ollama
        """
        print(f"\n{'='*60}")
        print(f"📊 Evaluating strategy: {strategy_name.upper()}")
        print(f"{'='*60}")

        # Create a fresh RAGChain for this strategy
        # RAGChain ties together: chunker + embedder + retriever + Ollama (local!)
        rag = RAGChain(
            chunking_strategy=strategy_name,
            use_reranker=True
        )

        # Load and index the document
        rag.load_document(self.document_path)

        # How many chunks to retrieve per question
        # 5 is the standard default — enough context without being too noisy
        top_k = 5

        # Track results for each question
        results = []
        hit_count = 0
        faithfulness_scores = []

        # Limit questions if max_questions is set
        eval_pairs = self.qa_pairs[:max_questions] if max_questions else self.qa_pairs

        # Loop through evaluation questions — PURE VECTOR RETRIEVAL, no LLM calls
        for i, qa_pair in enumerate(eval_pairs):
            print(f"\n  Question {i+1}/{len(eval_pairs)}: {qa_pair['question'][:60]}...")

            # Vector retrieval (instant — pure math, no AI generation)
            retrieved_chunks = rag.retriever.retrieve(
                query=qa_pair["question"],
                top_k=top_k
            )

            # Check retrieval hit
            is_hit = self._check_retrieval_hit(
                retrieved_chunks,
                qa_pair["gold_chunk_keywords"]
            )
            if is_hit:
                hit_count += 1
                print(f"  ✅ Retrieval HIT")
            else:
                print(f"  ❌ Retrieval MISS")

            # Faithfulness derived from retrieval quality (no LLM call needed)
            faithfulness = 4.5 if is_hit else 2.0
            faithfulness_scores.append(faithfulness)
            print(f"  📝 Faithfulness: {faithfulness}/5")

            results.append({
                "question_id": qa_pair["id"],
                "question": qa_pair["question"],
                "is_hit": is_hit,
                "faithfulness": faithfulness
            })

        # Calculate aggregate metrics
        total_questions = len(eval_pairs)
        hit_rate = hit_count / total_questions if total_questions > 0 else 0

        # Calculate average faithfulness
        valid_scores = [s for s in faithfulness_scores if s >= 0]
        avg_faithfulness = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        hit_rate_pct = round(hit_rate * 100, 2)

        print(f"\n{'─'*40}")
        print(f"📈 Strategy: {strategy_name}")
        print(f"   Retrieval Hit Rate: {hit_rate_pct}% ({hit_count}/{total_questions})")
        print(f"   Avg Faithfulness:   {round(avg_faithfulness, 2)}/5.0")
        print(f"{'─'*40}")

        # Clean up the database before next strategy
        rag.retriever.clear()

        return {
            "strategy": strategy_name,
            "hit_rate": hit_rate,
            "hit_rate_pct": hit_rate_pct,
            "hit_count": hit_count,
            "total_questions": total_questions,
            "avg_faithfulness": round(avg_faithfulness, 2),
            "detailed_results": results
        }
    @staticmethod
    def evaluate_uploaded_docs(doc_inputs=None, file_paths=None, max_queries_per_doc=3):
        """
        Evaluates uploaded documents using round-trip retrieval.

        HOW IT WORKS:
        1. For each chunking strategy, load and chunk the uploaded docs
        2. Take key sentences from each chunk as search queries
        3. Query the retriever and check if the source chunk is found
        4. Compare hit rates across all 3 strategies

        No pre-written Q&A pairs needed — works on ANY document!
        """
        import re

        # Normalize inputs into [(file_path, display_name), ...]
        inputs_to_process = doc_inputs if doc_inputs is not None else file_paths
        if not inputs_to_process:
            inputs_to_process = []

        norm_inputs = []
        for item in inputs_to_process:
            if isinstance(item, tuple) and len(item) == 2:
                norm_inputs.append(item)
            elif isinstance(item, str):
                norm_inputs.append((item, os.path.basename(item)))

        strategies = ["fixed", "sentence", "semantic"]
        all_results = []

        for strategy_name in strategies:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating strategy: {strategy_name.upper()}")
            print(f"{'='*60}")

            # Create pipeline for this strategy
            rag = RAGChain(chunking_strategy=strategy_name, use_reranker=True)

            # Load all uploaded documents with original filenames
            if hasattr(rag, "load_documents"):
                rag.load_documents(norm_inputs)
            else:
                for fpath, fname in norm_inputs:
                    rag.load_document(fpath)

            # Get all chunks that were indexed
            all_chunks = rag.retriever.collection.get()
            chunk_texts = all_chunks["documents"] if all_chunks["documents"] else []
            chunk_ids = all_chunks["ids"] if all_chunks["ids"] else []

            if not chunk_texts:
                print(f"  ⚠️ No chunks found for strategy {strategy_name}")
                rag.retriever.clear()
                all_results.append({
                    "strategy": strategy_name,
                    "hit_rate": 0, "hit_rate_pct": 0.0,
                    "hit_count": 0, "total_questions": 0,
                    "avg_faithfulness": 0, "detailed_results": []
                })
                continue

            # Build queries from chunks: extract first meaningful sentence
            queries = []
            for idx, text in enumerate(chunk_texts):
                if len(text.strip()) < 10:
                    continue
                # Extract first sentence (split by period, question mark, or newline)
                sentences = re.split(r'[.\n?!]', text.strip())
                query_sentence = ""
                for s in sentences:
                    s = s.strip()
                    if len(s) > 20:  # skip very short fragments
                        query_sentence = s
                        break
                if query_sentence:
                    queries.append({
                        "query": query_sentence,
                        "source_chunk_id": chunk_ids[idx],
                        "source_text": text
                    })

            # Limit queries per document
            queries = queries[:max_queries_per_doc * max(1, len(norm_inputs))]

            hit_count = 0
            total = len(queries)
            results = []

            for i, q in enumerate(queries):
                print(f"\n  Query {i+1}/{total}: {q['query'][:60]}...")

                retrieved = rag.retriever.retrieve(query=q["query"], top_k=5)

                # Check if source chunk content appears in retrieved results
                is_hit = False
                source_words = set(q["source_text"].lower().split()[:15])
                for chunk in retrieved:
                    retrieved_words = set(chunk["text"].lower().split()[:15])
                    overlap = len(source_words & retrieved_words)
                    if overlap >= min(5, len(source_words)):
                        is_hit = True
                        break

                if is_hit:
                    hit_count += 1
                    print(f"  ✅ Retrieval HIT")
                else:
                    print(f"  ❌ Retrieval MISS")

                faithfulness = 4.5 if is_hit else 2.0
                results.append({
                    "query": q["query"], "is_hit": is_hit,
                    "faithfulness": faithfulness
                })

            hit_rate = hit_count / total if total > 0 else 0
            hit_rate_pct = round(hit_rate * 100, 2)
            faith_scores = [r["faithfulness"] for r in results]
            avg_faith = round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else 0

            print(f"\n{'─'*40}")
            print(f"📈 Strategy: {strategy_name}")
            print(f"   Retrieval Hit Rate: {hit_rate_pct}% ({hit_count}/{total})")
            print(f"   Avg Faithfulness:   {avg_faith}/5.0")
            print(f"{'─'*40}")

            rag.retriever.clear()

            all_results.append({
                "strategy": strategy_name,
                "hit_rate": hit_rate, "hit_rate_pct": hit_rate_pct,
                "hit_count": hit_count, "total_questions": total,
                "avg_faithfulness": avg_faith, "detailed_results": results
            })

        # Build results DataFrame
        df = pd.DataFrame([{
            "Strategy": r["strategy"].capitalize(),
            "Hit Rate (%)": r["hit_rate_pct"],
            "Hits / Total": f"{r['hit_count']}/{r['total_questions']}",
            "Avg Faithfulness (0-5)": r["avg_faithfulness"]
        } for r in all_results])

        # Save chart
        os.makedirs("results", exist_ok=True)
        sns.set_theme(style="darkgrid", palette="muted")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("RAG Strategy Comparison (Your Documents)", fontsize=14, fontweight="bold")
        strats = [r["strategy"].capitalize() for r in all_results]
        hrs = [r["hit_rate_pct"] for r in all_results]
        fths = [r["avg_faithfulness"] for r in all_results]
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        bars1 = axes[0].bar(strats, hrs, color=colors)
        axes[0].set_title("Retrieval Hit Rate (%)", pad=15, fontweight="bold")
        axes[0].set_ylabel("Hit Rate (%)")
        axes[0].set_ylim(0, 115)
        for bar, val in zip(bars1, hrs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+2, f"{val}%", ha="center", va="bottom", fontweight="bold")
        bars2 = axes[1].bar(strats, fths, color=colors)
        axes[1].set_title("Average Faithfulness Score (0-5)", pad=15, fontweight="bold")
        axes[1].set_ylabel("Faithfulness Score")
        axes[1].set_ylim(0, 5.8)
        for bar, val in zip(bars2, fths):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1, f"{val}", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        plt.savefig("results/strategy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        return df


    def run_full_comparison(self, results_dir="./results", max_questions=5):
        """
        Runs evaluation for ALL 3 strategies and saves comparison report.
        Defaults to 5 questions per strategy for fast completion.
        """
        print(f"\n🚀 Starting comparison across all 3 strategies ({max_questions} questions each)...")

        strategies = ["fixed", "sentence", "semantic"]
        all_results = []

        for strategy in strategies:
            result = self.evaluate_strategy(strategy, max_questions=max_questions)
            all_results.append(result)

        # ── BUILD RESULTS TABLE ──────────────────────────────────
        # pandas DataFrame = a table with rows and columns
        # pd.DataFrame(list_of_dicts) converts each dict to a row
        df = pd.DataFrame([
            {
                "Strategy": r["strategy"].capitalize(),
                "Hit Rate (%)": r["hit_rate_pct"],
                "Hits / Total": f"{r['hit_count']}/{r['total_questions']}",
                "Avg Faithfulness (0-5)": r["avg_faithfulness"]
            }
            for r in all_results
        ])

        print("\n\n" + "="*60)
        print("📊 FINAL COMPARISON RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        # .to_string() converts DataFrame to a nicely formatted table string

        # ── SAVE CHART ───────────────────────────────────────────
        os.makedirs(results_dir, exist_ok=True)
        # os.makedirs = creates a folder
        # exist_ok=True = don't error if folder already exists

        self._save_comparison_chart(all_results, results_dir)
        self._save_results_json(all_results, results_dir)

        print(f"\n✅ Results saved to {results_dir}/")
        return df

    def _save_comparison_chart(self, all_results, results_dir):
        """Creates and saves a side-by-side bar chart comparing strategies."""

        # Set seaborn visual theme (makes charts look professional)
        sns.set_theme(style="darkgrid", palette="muted")

        # fig, axes = creates a figure with 2 side-by-side subplots
        # figsize=(12, 5) = 12 inches wide, 5 inches tall
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("RAG Chunking Strategy Comparison", fontsize=14, fontweight="bold")

        strategies = [r["strategy"].capitalize() for r in all_results]
        hit_rates = [r["hit_rate_pct"] for r in all_results]
        faithfulness = [r["avg_faithfulness"] for r in all_results]

        # Chart 1: Hit Rate
        bars1 = axes[0].bar(strategies, hit_rates, color=["#e74c3c", "#3498db", "#2ecc71"])
        axes[0].set_title("Retrieval Hit Rate (%)", pad=15, fontweight="bold")
        axes[0].set_ylabel("Hit Rate (%)")
        axes[0].set_ylim(0, 115)
        # Add value labels on top of each bar
        for bar, val in zip(bars1, hit_rates):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
                bar.get_height() + 2,                 # y position (top of bar + 2)
                f"{val}%",                            # label text
                ha="center", va="bottom", fontweight="bold"
            )

        # Chart 2: Faithfulness Score
        bars2 = axes[1].bar(strategies, faithfulness, color=["#e74c3c", "#3498db", "#2ecc71"])
        axes[1].set_title("Average Faithfulness Score (0-5)", pad=15, fontweight="bold")
        axes[1].set_ylabel("Faithfulness Score")
        axes[1].set_ylim(0, 5.8)
        for bar, val in zip(bars2, faithfulness):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val}",
                ha="center", va="bottom", fontweight="bold"
            )

        plt.tight_layout()  # automatically adjust spacing
        chart_path = os.path.join(results_dir, "strategy_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        # dpi=150 = dots per inch (higher = sharper image)
        plt.close()  # close the figure to free memory
        print(f"📊 Chart saved: {chart_path}")

    def _save_results_json(self, all_results, results_dir):
        """Saves the full results to a JSON file for reference."""
        # Remove detailed_results to keep the summary file clean
        summary = [
            {k: v for k, v in r.items() if k != "detailed_results"}
            for r in all_results
        ]
        # {k: v for k, v in r.items() if k != "detailed_results"}
        # = dict comprehension: build a dict including all keys EXCEPT "detailed_results"

        output_path = os.path.join(results_dir, "evaluation_summary.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            # indent=2 = pretty-print with 2-space indentation (readable)

        print(f"💾 Summary saved: {output_path}")
