#!/usr/bin/env python3
"""
Prepare pretraining datasets for HelixLM models.

Creates model-size-relevant subsets of:
  - HuggingFaceFW/fineweb-edu   (primary pretraining)
  - open-web-math/open-web-math  (math/reasoning boost, optional)

Math ratio is expressed as a percentage of the PRETRAINING budget.
Default: 1% math / 99% FineWeb-Edu.

Usage:
    python prepare_helixlm_pretrain.py --model-size tiny
    python prepare_helixlm_pretrain.py --model-size small --hf-org myorg --push-to-hub
    python prepare_helixlm_pretrain.py --model-size tiny --total-tokens 5000000
    python prepare_helixlm_pretrain.py --model-size base --fineweb-subset sample-100BT
    python prepare_helixlm_pretrain.py --model-size tiny --no-include-math
    python prepare_helixlm_pretrain.py --model-size base --math-ratio 0.005
"""

import argparse
import gc
import os
import sys
import json
import random
from math import ceil
from typing import Optional, Dict, List, Any, Iterator, Tuple
from datetime import datetime

from datasets import load_dataset, Dataset, DatasetDict, Features, Value

HF_TOKEN os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

PRETRAIN_DATASETS = {
    "fineweb-edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_column": "text",
        "description": "Web text filtered for educational quality",
    },
    "open-web-math": {
        "name": "open-web-math/open-web-math",
        "subset": None,
        "text_column": "text",
        "description": "Math-focused web text for reasoning boost",
    },
}

MODEL_SIZE_TOKENS = {
    "tiny":      5_000_000,
    "small":     50_000_000,
    "base":      250_000_000,
    "medium":    1_000_000_000,
    "large":     3_000_000_000,
    "xl":        10_000_000_000,
    "xxl":       40_000_000_000,
}

TOKENS_PER_SAMPLE = {
    "fineweb-edu": 500,
    "open-web-math": 800,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_dataset_name(
    model_size: str,
    total_tokens: int,
    pretrain_samples: int,
    hf_org: Optional[str] = None,
) -> str:
    size_label = model_size.lower()
    tokens_m = total_tokens / 1_000_000
    name = f"HelixLM-{size_label}"
    name += f"-{tokens_m:.1f}Mt"
    name += f"-{pretrain_samples}pt"
    name += f"-{datetime.now().strftime('%Y%m%d')}"
    if hf_org:
        return f"{hf_org}/{name}"
    return name


def stream_dataset_texts(
    dataset_name: str,
    subset: Optional[str],
    num_samples: int,
    text_column: Optional[str] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
) -> Iterator[str]:
    print(f"    Streaming from {dataset_name}" + (f" (subset={subset})" if subset else ""))
    ds = load_dataset(
        dataset_name,
        subset,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    count = 0
    for sample in ds:
        if count >= num_samples:
            break

        text = sample.get(text_column, "") if text_column else ""
        if isinstance(text, str) and text.strip():
            yield text.strip()
            count += 1

    print(f"    Collected {count:,} samples")


def stream_from_shards(
    dataset_name: str,
    num_samples: int,
    seed: int = 42,
) -> Iterator[str]:
    from huggingface_hub import HfApi
    import pyarrow.parquet as pq

    api = HfApi()
    print(f"    Discovering shards for {dataset_name} ...")
    repo_files = list(api.list_repo_files(dataset_name, repo_type="dataset"))
    parquet_files = [f for f in repo_files if f.endswith(".parquet") and "/data/" in f]

    if not parquet_files:
        print(f"    WARNING: No parquet shards found. Falling back to standard streaming.")
        yield from stream_dataset_texts(dataset_name, None, num_samples, "text", seed=seed)
        return

    print(f"    Found {len(parquet_files)} parquet shards")
    random.seed(seed)
    random.shuffle(parquet_files)

    collected = 0
    for shard_path in parquet_files:
        if collected >= num_samples:
            break
        try:
            local_path = api.hf_hub_download(
                dataset_name,
                shard_path,
                repo_type="dataset",
                local_dir=os.path.join(os.getcwd(), "_temp_shards"),
                local_dir_use_symlinks=False,
            )
            table = pq.read_table(local_path)
            rows = table.to_pylist()
            random.shuffle(rows)

            for row in rows:
                if collected >= num_samples:
                    break
                text = row.get("text", "")
                if isinstance(text, str) and text.strip():
                    yield text.strip()
                    collected += 1

            del table, rows
            gc.collect()
        except Exception as e:
            print(f"    WARNING: Failed to process shard {shard_path}: {e}")
            continue

    print(f"    Collected {collected:,} samples from shards")


def create_dataset_dict(
    model_size: str,
    total_tokens: Optional[int] = None,
    fineweb_subset: str = "sample-10BT",
    use_shards: bool = False,
    val_split: float = 0.02,
    seed: int = 42,
    shuffle_buffer: int = 10000,
    include_math: bool = True,
    math_ratio: float = 0.01,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    random.seed(seed)

    if total_tokens is None:
        total_tokens = MODEL_SIZE_TOKENS.get(model_size, MODEL_SIZE_TOKENS["tiny"])

    if not (0.0 <= math_ratio <= 1.0):
        raise ValueError(f"math_ratio must be in [0.0, 1.0], got {math_ratio}")

    print(f"\n{'='*60}")
    print(f"HelixLM Pretraining Dataset Preparation")
    print(f"{'='*60}")
    print(f"Model size: {model_size}")
    print(f"Total token budget: {total_tokens:,}")
    print(f"FineWeb-Edu subset: {fineweb_subset}")
    print(f"Include math: {include_math}")
    if include_math:
        print(f"Math ratio: {math_ratio:.2%} of the pretraining budget")
    print(f"Shard mode: {use_shards}")
    print(f"{'='*60}\n")

    if include_math:
        math_tokens = int(total_tokens * math_ratio)
        fineweb_tokens = total_tokens - math_tokens
    else:
        math_tokens = 0
        fineweb_tokens = total_tokens

    fineweb_samples = max(1, ceil(fineweb_tokens / TOKENS_PER_SAMPLE["fineweb-edu"])) if fineweb_tokens > 0 else 0
    openwebmath_samples = max(1, ceil(math_tokens / TOKENS_PER_SAMPLE["open-web-math"])) if (include_math and math_tokens > 0) else 0

    print("Dataset sampling plan:")
    print(f"  FineWeb-Edu:    {fineweb_samples:,} samples (~{fineweb_tokens:,} tokens)")
    if include_math:
        print(f"  OpenWebMath:    {openwebmath_samples:,} samples (~{math_tokens:,} tokens)")
    else:
        print(f"  OpenWebMath:    SKIPPED")
    print()

    # Pretraining data
    print("Streaming pretraining data...")
    pretrain_texts = []
    pretrain_sources = []

    print("  Loading FineWeb-Edu...")
    if use_shards and fineweb_subset == "full":
        fw_iter = stream_from_shards(
            PRETRAIN_DATASETS["fineweb-edu"]["name"],
            fineweb_samples,
            seed=seed,
        )
    else:
        fw_iter = stream_dataset_texts(
            PRETRAIN_DATASETS["fineweb-edu"]["name"],
            fineweb_subset if fineweb_subset else PRETRAIN_DATASETS["fineweb-edu"]["subset"],
            fineweb_samples,
            text_column=PRETRAIN_DATASETS["fineweb-edu"]["text_column"],
            shuffle_buffer=shuffle_buffer,
            seed=seed,
        )

    for text in fw_iter:
        pretrain_texts.append(text)
        pretrain_sources.append("fineweb-edu")
    print(f"    FineWeb-Edu: {len(pretrain_texts):,} samples\n")

    if include_math and openwebmath_samples > 0:
        print("  Loading OpenWebMath...")
        owm_iter = stream_dataset_texts(
            PRETRAIN_DATASETS["open-web-math"]["name"],
            PRETRAIN_DATASETS["open-web-math"]["subset"],
            openwebmath_samples,
            text_column=PRETRAIN_DATASETS["open-web-math"]["text_column"],
            shuffle_buffer=shuffle_buffer,
            seed=seed + 1,
        )

        owm_count = 0
        for text in owm_iter:
            pretrain_texts.append(text)
            pretrain_sources.append("open-web-math")
            owm_count += 1
        print(f"    OpenWebMath: {owm_count:,} samples")
    else:
        owm_count = 0
        print("  Skipping OpenWebMath.\n")

    print(f"  Total pretraining: {len(pretrain_texts):,} samples\n")

    # Shuffle and split
    print("Shuffling and creating train/val splits...")

    indices = list(range(len(pretrain_texts)))
    random.shuffle(indices)
    split_idx = int(len(pretrain_texts) * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    features = Features({
        "text": Value("string"),
        "source": Value("string"),
    })

    dataset_dict = DatasetDict({
        "pretrain_train": Dataset.from_dict({
            "text": [pretrain_texts[i] for i in train_idx],
            "source": [pretrain_sources[i] for i in train_idx],
        }, features=features),
        "pretrain_val": Dataset.from_dict({
            "text": [pretrain_texts[i] for i in val_idx],
            "source": [pretrain_sources[i] for i in val_idx],
        }, features=features),
    })

    used_ratios = {
        "fineweb-edu": 1.0 - math_ratio if include_math else 1.0,
        "open-web-math": math_ratio if include_math else 0.0,
    }

    metadata = {
        "model_size": model_size,
        "total_tokens": total_tokens,
        "token_ratios": used_ratios,
        "include_math": include_math,
        "math_ratio_of_pretrain": math_ratio if include_math else 0.0,
        "fineweb_edu_samples": len([s for s in pretrain_sources if s == "fineweb-edu"]),
        "openwebmath_samples": len([s for s in pretrain_sources if s == "open-web-math"]),
        "pretrain_train_samples": len(train_idx),
        "pretrain_val_samples": len(val_idx),
        "pretrain_total_samples": len(pretrain_texts),
        "val_split": val_split,
        "fineweb_subset": fineweb_subset,
        "use_shards": use_shards,
        "seed": seed,
        "created": datetime.now().isoformat(),
    }

    print(f"\nDataset splits:")
    print(f"  pretrain_train:  {len(train_idx):,} samples")
    print(f"  pretrain_val:    {len(val_idx):,} samples")

    return dataset_dict, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare HelixLM pretraining datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-size tiny
  %(prog)s --model-size small --hf-org myorg --push-to-hub
  %(prog)s --model-size tiny --total-tokens 10000000 --fineweb-subset sample-100BT
  %(prog)s --model-size base --use-shards --fineweb-subset full
  %(prog)s --model-size tiny --no-include-math
  %(prog)s --model-size small --math-ratio 0.005
        """,
    )
    parser.add_argument("--model-size", type=str, default="tiny", choices=list(MODEL_SIZE_TOKENS.keys()))
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--fineweb-subset", type=str, default="sample-10BT", choices=["sample-10BT", "sample-100BT", "sample-350BT", "full"])
    parser.add_argument("--use-shards", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-math", action="store_true", default=True, help="Include OpenWebMath (default: True)")
    parser.add_argument("--no-include-math", action="store_false", dest="include_math", help="Exclude OpenWebMath; use only FineWeb-Edu")
    parser.add_argument("--math-ratio", type=float, default=0.01, help="Fraction of the pretraining budget for OpenWebMath (default: 0.01 = 1%%)")
    parser.add_argument("--hf-org", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./helixlm-datasets")
    parser.add_argument("--save-local", action="store_true", default=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_dict, metadata = create_dataset_dict(
        model_size=args.model_size,
        total_tokens=args.total_tokens,
        fineweb_subset=args.fineweb_subset,
        use_shards=args.use_shards,
        val_split=args.val_split,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        include_math=args.include_math,
        math_ratio=args.math_ratio,
    )

    repo_name = args.hf_repo or generate_dataset_name(
        model_size=args.model_size,
        total_tokens=metadata["total_tokens"],
        pretrain_samples=metadata["pretrain_total_samples"],
        hf_org=args.hf_org,
    )

    local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
    local_path = os.path.join(args.output_dir, local_name)
    metadata["repo_name"] = repo_name
    metadata["local_path"] = local_path

    if args.save_local:
        print(f"\nSaving dataset locally to: {local_path}")
        dataset_dict.save_to_disk(local_path)
        metadata_path = os.path.join(local_path, "dataset_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    if args.push_to_hub:
        print(f"\nPushing dataset to HuggingFace Hub: {repo_name}")
        dataset_dict.push_to_hub(
            repo_name,
            private=False,
            commit_message=f"HelixLM {args.model_size} pretraining dataset ({metadata['total_tokens']:,} tokens)",
            token=HF_TOKEN
        )
        print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_name}")

    print(f"\n{'='*60}")
    print(f"Dataset Preparation Complete")
    print(f"{'='*60}")
    print(f"Repo name: {repo_name}")
    print(f"Local path: {local_path}")
    print(f"Total tokens (target): {metadata['total_tokens']:,}")
    print(f"Pretraining: {metadata['pretrain_total_samples']:,} ({metadata['pretrain_train_samples']:,} train / {metadata['pretrain_val_samples']:,} val)")
    print(f"{'='*60}\n")
    print("METADATA_JSON_START")
    print(json.dumps(metadata))
    print("METADATA_JSON_END")


if __name__ == "__main__":
    main()
