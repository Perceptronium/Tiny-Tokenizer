import argparse
import pickle
from pathlib import Path
import sys
import json

from models.tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        prog="test_tokenizer",
        description="Encode or decode text with a BPE tokenizer."
    )
    p.add_argument(
        "--vocab", required=True, type=Path,
        help="Path to vocab pickle (id -> bytes)."
    )
    p.add_argument(
        "--merges", required=True, type=Path,
        help="Path to merges pickle (list of (bytes, bytes))."
    )
    p.add_argument(
        "--special-token", action="append", default=[],
        help="Special token string to reserve (repeat flag for multiple)."
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--encode", type=str,
        help="Raw text to encode."
    )

    g.add_argument(
        "--decode", type=str,
        help="IDs to decode; accepts space/comma-separated numbers OR a JSON list."
    )
    return p.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def read_text_input(args):
    if args.encode is not None:
        return args.encode
    raise ValueError("No encode input provided.")


def read_ids_input(args):
    raw: str
    if args.decode is not None:
        raw = args.decode
    elif args.decode_file is not None:
        raw = args.decode_file.read_text(encoding="utf-8")
    else:
        raise ValueError("No decode input provided.")

    raw = raw.strip()
    try:
        val = json.loads(raw)
        if isinstance(val, list) and all(isinstance(x, int) for x in val):
            return val
    except Exception:
        pass

    tokens = [t for t in raw.replace(",", " ").split() if t]
    try:
        return [int(t) for t in tokens]
    except ValueError as e:
        raise ValueError(
            "Could not parse IDs. Provide a JSON list like [1,2,3] "
            "or a comma/space-separated list like '1, 2, 3'."
        ) from e


if __name__ == "__main__":
    args = parse_args()

    if not args.vocab.exists():
        print(f"Error: vocab not found: {args.vocab}", file=sys.stderr)
        sys.exit(1)
    if not args.merges.exists():
        print(f"Error: merges not found: {args.merges}", file=sys.stderr)
        sys.exit(1)

    vocab = load_pickle(args.vocab)
    merges = load_pickle(args.merges)

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=args.special_token or None
    )

    # Encode mode
    if args.encode is not None:
        text = read_text_input(args)
        ids = tokenizer.encode(text)
        print(json.dumps(ids, ensure_ascii=False))

    # Decode mode
    else:
        ids = read_ids_input(args)
        text = tokenizer.decode(ids)
        print(text, end="")
