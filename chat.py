"""
chat.py
-------
CLI chatbot for the Self-Correcting RAG system.

Usage:
    python chat.py              # interactive mode
    python chat.py --verbose    # also print the decision trace

For the full web chatbot UI, run:
    streamlit run app.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Correcting RAG CLI")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full decision trace after each answer.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Self-Correcting RAG  —  CLI Mode")
    print("  Type your question and press Enter.")
    print("  Type 'exit' or press Ctrl+C to quit.")
    if not args.verbose:
        print("  Tip: run with --verbose to see the decision trace.")
    print("=" * 60 + "\n")

    # Lazy import so startup errors (e.g. missing .env) are shown cleanly.
    try:
        from src.graph.correction_graph import run_corrected_query
    except ImportError as exc:
        print(f"[ERROR] Could not load RAG pipeline: {exc}")
        print("Make sure you have installed requirements and set up your .env file.")
        sys.exit(1)

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query or query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        print("\nThinking…\n")
        try:
            result = run_corrected_query(query)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Pipeline failed: {exc}\n")
            continue

        answer = result.get("final_answer", "").strip()
        correction_rounds = result.get("correction_rounds", 0)
        reretrieval = result.get("reretrieval_happened", False)

        print("─" * 60)
        print(f"Answer:\n{answer}")
        print("─" * 60)

        # Quick stats
        stats = []
        if correction_rounds:
            stats.append(f"corrected {correction_rounds}×")
        if reretrieval:
            stats.append("re-retrieval ran")
        if stats:
            print(f"[Pipeline: {', '.join(stats)}]")

        if args.verbose:
            trace = result.get("trace", [])
            if trace:
                print("\nDecision trace:")
                for i, entry in enumerate(trace, 1):
                    print(f"  {i}. [{entry.get('node', '?')}] {entry.get('decision', '')}")

        print()


if __name__ == "__main__":
    main()
