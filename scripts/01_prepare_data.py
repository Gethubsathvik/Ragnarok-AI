"""Step 2: download text, GPT-2 tokenize, split 80/20, write binary token files.

Default source: WikiText-103 (raw) ~ 540 MB of Wikipedia text (~110M GPT-2 tokens).
Use --source local --local-dir path/to/txt_files for your own public-domain books.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from config import DATA_DIR, TOKENIZER_DIR
from dataset import make_loaders

TITLE_RE = re.compile(r"^ = [^=].*[^=] = \n?$")  # top-level wikitext article title


def load_wikitext():
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    docs, cur = [], []
    for split in ("train", "validation", "test"):
        for line in ds[split]["text"]:
            if TITLE_RE.match(line) and cur:
                docs.append("".join(cur))
                cur = []
            cur.append(line)
    if cur:
        docs.append("".join(cur))
    return docs


def load_local(folder, chunk_chars=20000):
    docs = []
    for p in sorted(Path(folder).rglob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i in range(0, len(text), chunk_chars):  # split long books into chunks
            docs.append(text[i : i + chunk_chars])
    return docs


def tokenize(docs, tok, path, batch=256):
    """Tokenize and append to `path` chunk by chunk (constant memory). Returns the token count."""
    total = 0
    with open(path, "wb") as f:
        for i in tqdm(range(0, len(docs), batch), desc=f"tokenizing -> {Path(path).name}"):
            chunk = []
            for ids in tok(docs[i : i + batch])["input_ids"]:
                chunk.extend(ids)
                chunk.append(tok.eos_token_id)  # <|endoftext|> between documents
            arr = np.asarray(chunk, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(arr)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wikitext", "local"], default="wikitext")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--max-mb", type=float, default=1000, help="cap raw text size (MB)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4, help="only for the demo batch")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.source == "local" and not args.local_dir:
        ap.error("--source local requires --local-dir")

    DATA_DIR.mkdir(exist_ok=True)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10**9
    tok.save_pretrained(TOKENIZER_DIR)  # cached so training/GUI work offline

    print(f"Loading text source: {args.source}")
    docs = load_wikitext() if args.source == "wikitext" else load_local(args.local_dir)
    random.Random(args.seed).shuffle(docs)

    kept, total = [], 0
    for d in docs:
        total += len(d.encode("utf-8"))
        if total > args.max_mb * 1e6:
            break
        kept.append(d)
    docs = kept
    print(f"{len(docs):,} documents, {total / 1e6:.0f} MB raw text")

    n_train = int(len(docs) * args.train_frac)
    splits = {"train": docs[:n_train], "val": docs[n_train:]}
    meta = {"vocab": tok.vocab_size, "eos": tok.eos_token_id}
    for name, d in splits.items():
        n_tok = tokenize(d, tok, DATA_DIR / f"{name}.bin")
        meta[f"{name}_tokens"] = int(n_tok)
        print(f"{name}: {len(d):,} docs -> {n_tok:,} tokens")
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---------- show samples to verify ----------
    train = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    print("\n=== SAMPLE: first 40 token ids ===")
    print(train[:40].tolist())
    print("\n=== SAMPLE: decoded back to text ===")
    print(repr(tok.decode(train[:60].tolist())))
    print("\n=== SAMPLE: token-by-token (id -> piece) ===")
    print(" | ".join(f"{i}:{tok.decode([i])!r}" for i in train[:20].tolist()))
    text = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(text)
    print(f"\nRound trip: {text!r}\n  ids={ids}\n  back={tok.decode(ids)!r}")
    assert tok.decode(ids) == text

    train_loader, val_loader = make_loaders(args.batch_size, args.block_size, num_workers=0)
    x, y = next(iter(train_loader))
    assert (x[:, 1:] == y[:, :-1]).all(), "targets must be inputs shifted by one"
    print(f"\n=== DATALOADER ===\ntrain batches: {len(train_loader):,} | val batches: {len(val_loader):,}")
    print(f"x shape {tuple(x.shape)}, y shape {tuple(y.shape)}, dtype {x.dtype}")
    print("x[0][:12]:", x[0][:12].tolist())
    print("y[0][:12]:", y[0][:12].tolist(), "(shifted by one)")
    print("\nData ready.")


if __name__ == "__main__":
    main()
