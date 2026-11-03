"""CLI entry: batch eval. Default: idle keep-alive for Docker."""

import argparse
import time

from qdrant_client import QdrantClient

from config import QDRANT_URL
from eval import default_reporters, run_batch
from loaders import load_dialogs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dialogs",
        metavar="PATH",
        help="eval input: dialogs .yaml (default data/dialogs.yaml)",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="eval: also write data/out/answer.xlsx (default: console + Langfuse only)",
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help="eval: also write data/out/answer_<stamp>.txt",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="run batch eval against the existing Qdrant collection",
    )
    return parser.parse_args()


def build_reporters(args):
    """Assemble pluggable eval sinks from CLI flags."""
    return default_reporters(excel=args.excel, txt=args.txt)


def main():
    args = parse_args()
    # Default: idle so `docker compose up` does not burn GPU / LLM on eval.
    if not (args.eval or args.dialogs or args.excel or args.txt):
        print("llm service idle; pass --eval to run batch Q&A")
        while True:
            time.sleep(3600)

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    run_batch(
        client,
        dialogs=load_dialogs(args.dialogs),
        reporters=build_reporters(args),
    )


if __name__ == "__main__":
    main()
