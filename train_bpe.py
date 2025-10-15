import argparse
import pickle
import time
from pathlib import Path

from models.byte_pair_encoder import BytePairEncoder


def parse_args():
    p = argparse.ArgumentParser(
        prog="train_bpe",
        description="Train a Byte-Pair Encoding (BPE) tokenizer on a text corpus."
    )

    p.add_argument(
        "-d", "--data",
        required=True,
        type=Path,
        help="Path to the training text file."
    )

    p.add_argument(
        "-v", "--vocab-size",
        required=True,
        type=int,
        help="Target final vocabulary size (includes 256 bytes + any special tokens)."
    )

    # You can pass multiple --special-token flags (e.g. --special-token '<|endoftext|>' --special-token '<|fim_prefix|>')
    p.add_argument(
        "--special-token",
        action="append",
        default=['<|endoftext|>'],
        help="A special token to reserve (repeat this flag for multiple)."
    )

    # Or load specials from a file (one token per line)
    p.add_argument(
        "--special-tokens-file",
        type=Path,
        default=None,
        help="Optional path to a file listing special tokens (one per line)."
    )

    p.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to save the trained vocab/merges pickles. Will be created if missing."
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for outputs (default: derived from data filename)."
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information during training."
    )
    return p.parse_args()


def load_specials(args) -> list[str]:
    specials = list(args.special_token or [])
    if args.special_tokens_file:
        with args.special_tokens_file.open("r", encoding="utf-8") as f:
            file_specials = [line.rstrip("\n") for line in f if line.strip()]
        specials.extend(file_specials)
    seen = set()
    deduped = []
    for tok in specials:
        if tok not in seen:
            deduped.append(tok)
            seen.add(tok)
    return deduped


if __name__ == "__main__":

    args = parse_args()
    data_path: Path = args.data

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    special_tokens = load_specials(args)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default prefix from input filename if not provided
    prefix = args.prefix or data_path.stem

    # Filenames
    vocab_out = output_dir / f"{prefix}_bpe_vocab.pkl"
    merges_out = output_dir / f"{prefix}_bpe_merges.pkl"

    # Train
    if args.verbose:
        print(f"Training BPE")
        print(f"  data:            {data_path}")
        print(f"  vocab_size:      {args.vocab_size}")
        print(f"  special_tokens:  {special_tokens or '[]'}")
        print(f"  output_dir:      {output_dir}")
        print(f"  outputs:         {vocab_out.name}, {merges_out.name}")

    bpe = BytePairEncoder()
    t0 = time.time()

    vocab, merges = bpe.train_bpe(
        input_path=str(data_path),
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        verbose=True
    )
    t1 = time.time()

    if args.verbose:
        print(f"Trained in {t1 - t0:.2f}s")
        print(f"Vocab size learned: {len(vocab)}")
        print(f"Merges learned:     {len(merges)}")

    # Save
    with vocab_out.open("wb") as fp:
        pickle.dump(vocab, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with merges_out.open("wb") as fp:
        pickle.dump(merges, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if args.verbose:
        print("Saved:")
