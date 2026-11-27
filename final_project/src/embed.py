"""Extract frozen mutant-minus-WT last-layer embeddings at the substituted position."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from settings import CACHE, DATA, MODEL_SPECS, selected_assays


def embed_assay(assay: dict, key: str, tokenizer, encoder, device: torch.device, batch_size: int, limit: int) -> None:
    assay_id = assay["id"]
    folder = DATA / "processed" / assay_id
    metadata = json.loads((folder / "metadata.json").read_text())
    sequence = metadata["sequence"]
    input_structure = metadata["three_di"] if key == "P" else "#" * len(sequence) if key == "Q" else ""
    tokens = [aa + state for aa, state in zip(sequence, input_structure, strict=True)] if input_structure else list(sequence)
    ids = tokenizer("".join(tokens), return_tensors="pt")["input_ids"][0]
    vocab = tokenizer.get_vocab()
    if len(ids) != len(sequence) + 2 or not all(vocab[token] == ids[i + 1] for i, token in enumerate(tokens)):
        raise ValueError(f"Invalid residue tokenization for {assay_id} / {key}")
    if ids[0] != tokenizer.cls_token_id or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Unexpected model special tokens")

    variants = pd.read_csv(folder / "variants.csv", usecols=["variant_id", "position", "wt", "alt"])
    if limit:
        variants = variants.head(limit)
    variant_ids = variants["variant_id"].to_numpy(dtype=str)
    signature = hashlib.sha256(("".join(tokens) + MODEL_SPECS[key][1]).encode()).hexdigest()
    dest = DATA / "embeddings" / ("pilot" if limit else "full") / assay_id / f"{key}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as saved:
            if np.array_equal(variant_ids, saved["variant_id"]) and saved["signature"].item() == signature:
                print(f"{assay['label']} {key}: using verified embedding cache", flush=True)
                return
    # Tensor index i is the biological 1-based position because the first token is CLS.
    positions = torch.tensor(variants["position"].to_numpy(), dtype=torch.long, device=device)
    wt_letters = [sequence[i - 1] for i in variants["position"]]
    if wt_letters != variants["wt"].tolist():
        raise ValueError(f"Wild-type lookup mismatch in {assay_id}")
    replacements = [alt + input_structure[i - 1] if input_structure else alt
                    for i, alt in zip(variants["position"], variants["alt"], strict=True)]
    alt_ids = torch.tensor([vocab[token] for token in replacements], dtype=torch.long, device=device)
    ids = ids.to(device=device)
    parts = []
    with torch.inference_mode():
        wild_type = encoder(input_ids=ids.unsqueeze(0)).last_hidden_state[0]
        for start in range(0, len(variants), batch_size):
            stop = min(start + batch_size, len(variants))
            batch_positions = positions[start:stop]
            batch_idx = torch.arange(stop - start, device=device)
            mutant = ids.expand(stop - start, -1).clone()
            mutant[batch_idx, batch_positions] = alt_ids[start:stop]
            embeddings = encoder(input_ids=mutant).last_hidden_state[batch_idx, batch_positions]
            parts.append((embeddings.float() - wild_type[batch_positions].float()).cpu().numpy())
            if start == 0 or stop == len(variants) or (start // batch_size) % 16 == 15:
                print(f"{assay['label']} {key}: {stop}/{len(variants)} variants", flush=True)
    delta = np.concatenate(parts)
    if not np.isfinite(delta).all() or np.any(np.linalg.norm(delta, axis=1) <= 0):
        raise ValueError(f"Degenerate embedding for {assay_id} / {key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_suffix(".tmp.npz")
    np.savez_compressed(staging, variant_id=variant_ids, embedding=delta, signature=signature)
    staging.replace(dest)
    print(f"Saved {dest.relative_to(DATA.parent)} ({len(delta)} × {delta.shape[1]})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assay", nargs="+")
    parser.add_argument("--encoder", choices=MODEL_SPECS, nargs="+", default=tuple(MODEL_SPECS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pilot", type=int, default=0, metavar="N")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.pilot < 0:
        parser.error("Batch size and thread count must be positive, and pilot size nonnegative")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    print(f"Embedding on {device} with PyTorch {torch.__version__}", flush=True)
    for key in args.encoder:
        directory: Path = CACHE / "models" / key
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        checkpoint = AutoModelForMaskedLM.from_pretrained(directory, local_files_only=True, weights_only=True)
        encoder = checkpoint.base_model.to(device).eval()
        for assay in selected_assays(args.assay):
            embed_assay(assay, key, tokenizer, encoder, device, args.batch_size, args.pilot)
        del checkpoint, encoder


if __name__ == "__main__":
    main()
