# ============================================================
# FILE: run_evaluation.py
# PURPOSE: A standalone script to run the full evaluation
#          from the command line (without the UI).
#
# HOW TO RUN:
#   python run_evaluation.py
#
# WHEN TO USE THIS vs. THE UI:
#   Use this script when you want to run the evaluation overnight
#   and come back to results in the morning.
#   Use the UI (app.py) for interactive Q&A demos.
# ============================================================

import os
from dotenv import load_dotenv
from src.evaluator import Evaluator


import sys

def main():
    print("=" * 60)
    print("  Advanced RAG Evaluation Suite")
    print("  Comparing: Fixed | Sentence | Semantic Chunking")
    print("=" * 60)

    # Allow passing file paths via command line arguments
    if len(sys.argv) > 1:
        doc_paths = sys.argv[1:]
    else:
        # Look for files in data/ directory
        doc_paths = []
        if os.path.exists("data"):
            doc_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith((".pdf", ".txt", ".docx"))]

    if not doc_paths:
        print("❌ No documents found to evaluate.")
        print("   Usage: python run_evaluation.py <path_to_doc1> <path_to_doc2> ...")
        return

    print(f"✅ Evaluating documents: {', '.join(doc_paths)}")

    # ── RUN EVALUATION ────────────────────────────────────────
    results_df = Evaluator.evaluate_uploaded_docs(file_paths=doc_paths)
    print("\n⚡ Starting evaluation using local Ollama (Llama 3.2)...")
    print("   This will take 5-15 minutes.")
    print("   Each of 30 questions × 3 strategies = 90 total retrievals")
    print("   Plus local Ollama calls for faithfulness grading\n")

    evaluator = Evaluator(
        eval_dataset_path=eval_path,
        document_path=doc_path
    )

    # run_full_comparison() evaluates all 3 strategies and saves results
    results_df = evaluator.run_full_comparison(results_dir="results")

    # ── PRINT FINAL SUMMARY ───────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print("\nResults saved to:")
    print("  results/evaluation_summary.json  ← raw metrics")
    print("  results/strategy_comparison.png  ← comparison chart")
    print("\nOpen the chart to see the visual comparison!")
    print("\n📋 Your resume bullet:")

    if len(results_df) > 0:
        best = results_df.loc[results_df["Hit Rate (%)"].idxmax()]
        worst = results_df.loc[results_df["Hit Rate (%)"].idxmin()]
        print(f"""
  "Built advanced RAG pipeline comparing 3 chunking strategies
   (fixed-size, sentence, semantic); {best['Strategy']} chunking
   improved retrieval hit rate from {worst['Hit Rate (%)']}%
   → {best['Hit Rate (%)']}% on a 30-question evaluation harness
   with LLM-graded faithfulness of {best['Avg Faithfulness (0-5)']}/5.0.
   Implemented cross-encoder re-ranking (ms-marco-MiniLM)."
        """)


# ── ENTRY POINT ───────────────────────────────────────────────────
# This runs only when you execute: python run_evaluation.py
# NOT when this file is imported by another module
if __name__ == "__main__":
    main()
