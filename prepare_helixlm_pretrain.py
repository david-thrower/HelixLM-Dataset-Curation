#!/usr/bin/env python3
"""
Prepare pretraining datasets for HelixLM models.

TRULY memory-efficient version using chunked writing.
Writes samples in chunks, processes train/val split using indices.

Creates model-size-relevant subsets of:
  - HuggingFaceFW/fineweb-edu   (primary pretraining)
  - open-web-math/open-web-math  (math/reasoning boost, optional)

Usage:
    python prepare_helixlm_pretrain_streaming.py --model-size tiny
    python prepare_helixlm_pretrain_streaming.py --model-size medium --total-tokens 1500000000
"""

import argparse
import gc
import os
import sys
import json
import random
import tempfile
import shutil
from math import ceil
from typing import Optional, Dict, List, Any, Iterator, Tuple
from datetime import datetime

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, concatenate_datasets

HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------

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

# Process in chunks to limit memory
CHUNK_SIZE = 25000  # samples per chunk


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
    text_column: str = "text",
    shuffle_buffer: int = 10000,
    seed: int = 42,
) -> Iterator[Tuple[str, str]]:
    """Yield (text, source) tuples."""
    print(f"    Streaming from {dataset_name}" + (f" (subset={subset})" if subset else ""))
    ds = load_dataset(
        dataset_name,
        subset,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    count = 0
    source_label = "fineweb-edu" if "fineweb" in dataset_name else "open-web-math"
    
    for sample in ds:
        if count >= num_samples:
            break

        text = sample.get(text_column, "")
        if isinstance(text, str) and text.strip():
            yield (text.strip(), source_label)
            count += 1

    print(f"    Collected {count:,} samples")


def create_dataset_chunked(
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
    """
    Memory-efficient dataset creation using chunked processing.
    
    Strategy:
    1. Stream samples from both sources
    2. Store in temporary chunked files (or memory-mapped structure)
    3. Shuffle indices deterministically
    4. Create train/val split by selecting appropriate chunks
    """
    random.seed(seed)

    if total_tokens is None:
        total_tokens = MODEL_SIZE_TOKENS.get(model_size, MODEL_SIZE_TOKENS["tiny"])

    if not (0.0 <= math_ratio <= 1.0):
        raise ValueError(f"math_ratio must be in [0.0, 1.0], got {math_ratio}")

    print(f"\n{'='*60}")
    print(f"HelixLM Pretraining Dataset Preparation (Chunked/Streaming)")
    print(f"{'='*60}")
    print(f"Model size: {model_size}")
    print(f"Total token budget: {total_tokens:,}")
    print(f"FineWeb-Edu subset: {fineweb_subset}")
    print(f"Include math: {include_math}")
    if include_math:
        print(f"Math ratio: {math_ratio:.2%} of the pretraining budget")
    print(f"Chunk size: {CHUNK_SIZE:,} samples")
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

    # Collect samples in chunks to disk
    temp_dir = tempfile.mkdtemp(prefix="helixlm_chunks_")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        features = Features({
            "text": Value("string"),
            "source": Value("string"),
        })
        
        # Stream and save in chunks
        all_chunk_files = []
        current_chunk = {"text": [], "source": []}
        chunk_num = 0
        total_samples = 0
        fw_samples = 0
        owm_samples = 0
        
        def save_chunk():
            nonlocal chunk_num, all_chunk_files
            if len(current_chunk["text"]) == 0:
                return
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num:05d}.arrow")
            ds = Dataset.from_dict(current_chunk, features=features)
            ds.save_to_disk(chunk_path)
            all_chunk_files.append(chunk_path)
            print(f"    Saved chunk {chunk_num}: {len(current_chunk['text']):,} samples")
            current_chunk["text"].clear()
            current_chunk["source"].clear()
            chunk_num += 1
            gc.collect()
        
        # Stream FineWeb-Edu
        print("Streaming FineWeb-Edu...")
        for text, source in stream_dataset_texts(
            PRETRAIN_DATASETS["fineweb-edu"]["name"],
            fineweb_subset if fineweb_subset else PRETRAIN_DATASETS["fineweb-edu"]["subset"],
            fineweb_samples,
            text_column=PRETRAIN_DATASETS["fineweb-edu"]["text_column"],
            seed=seed,
        ):
            current_chunk["text"].append(text)
            current_chunk["source"].append(source)
            fw_samples += 1
            total_samples += 1
            
            if len(current_chunk["text"]) >= CHUNK_SIZE:
                save_chunk()
        
        # Stream OpenWebMath
        if openwebmath_samples > 0:
            print("Streaming OpenWebMath...")
            for text, source in stream_dataset_texts(
                PRETRAIN_DATASETS["open-web-math"]["name"],
                PRETRAIN_DATASETS["open-web-math"]["subset"],
                openwebmath_samples,
                text_column=PRETRAIN_DATASETS["open-web-math"]["text_column"],
                seed=seed + 1,
            ):
                current_chunk["text"].append(text)
                current_chunk["source"].append(source)
                owm_samples += 1
                total_samples += 1
                
                if len(current_chunk["text"]) >= CHUNK_SIZE:
                    save_chunk()
        
        # Save final partial chunk
        save_chunk()
        
        print(f"\nTotal samples streamed: {total_samples:,}")
        print(f"  FineWeb-Edu: {fw_samples:,}")
        print(f"  OpenWebMath: {owm_samples:,}")
        print(f"  Chunks: {len(all_chunk_files)}")
        
        if total_samples == 0:
            raise ValueError("No samples collected!")
        
        # Create shuffled train/val indices
        print("\nCreating train/val split indices...")
        indices = np.arange(total_samples)
        np.random.seed(seed + 10)
        np.random.shuffle(indices)
        
        split_idx = int(total_samples * (1 - val_split))
        train_indices_set = set(indices[:split_idx].tolist())
        val_indices_set = set(indices[split_idx:].tolist())
        
        del indices
        gc.collect()
        
        print(f"  Train samples: {len(train_indices_set):,}")
        print(f"  Val samples: {len(val_indices_set):,}")
        
        # Load chunks and split into train/val
        print("\nProcessing chunks into train/val datasets...")
        train_chunks = []
        val_chunks = []
        global_idx = 0
        
        for chunk_file in sorted(all_chunk_files):
            chunk_ds = Dataset.load_from_disk(chunk_file)
            chunk_size = len(chunk_ds)
            
            # Determine which samples in this chunk go to train vs val
            chunk_train_indices = []
            chunk_val_indices = []
            
            for i in range(chunk_size):
                if global_idx + i in train_indices_set:
                    chunk_train_indices.append(i)
                else:
                    chunk_val_indices.append(i)
            
            if chunk_train_indices:
                train_chunk = chunk_ds.select(chunk_train_indices)
                train_chunks.append(train_chunk)
            
            if chunk_val_indices:
                val_chunk = chunk_ds.select(chunk_val_indices)
                val_chunks.append(val_chunk)
            
            del chunk_ds
            gc.collect()
            
            global_idx += chunk_size
            
            if chunk_file == all_chunk_files[-1] or global_idx % (CHUNK_SIZE * 4) == 0:
                print(f"  Processed {global_idx:,}/{total_samples:,} samples...")
        
        # Concatenate all chunks
        print("\nConcatenating chunks...")
        train_dataset = concatenate_datasets(train_chunks) if train_chunks else Dataset.from_dict({"text": [], "source": []}, features=features)
        val_dataset = concatenate_datasets(val_chunks) if val_chunks else Dataset.from_dict({"text": [], "source": []}, features=features)
        
        # Clean up chunks
        del train_chunks, val_chunks
        gc.collect()
        
        dataset_dict = DatasetDict({
            "pretrain_train": train_dataset,
            "pretrain_val": val_dataset,
        })
        
    finally:
        # Cleanup temp dir
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Metadata
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
        "fineweb_edu_samples": fw_samples,
        "openwebmath_samples": owm_samples,
        "pretrain_train_samples": len(train_dataset),
        "pretrain_val_samples": len(val_dataset),
        "pretrain_total_samples": total_samples,
        "val_split": val_split,
        "fineweb_subset": fineweb_subset,
        "use_shards": use_shards,
        "seed": seed,
        "created": datetime.now().isoformat(),
    }

    print(f"\nDataset splits:")
    print(f"  pretrain_train:  {len(train_dataset):,} samples")
    print(f"  pretrain_val:    {len(val_dataset):,} samples")

    return dataset_dict, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare HelixLM pretraining datasets (memory-efficient chunked)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-size tiny
  %(prog)s --model-size medium --total-tokens 1500000000
  %(prog)s --model-size small --hf-org myorg --push-to-hub
  %(prog)s --model-size base --fineweb-subset sample-100BT
        """,
    )
    parser.add_argument("--model-size", type=str, default="tiny", choices=list(MODEL_SIZE_TOKENS.keys()))
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--fineweb-subset", type=str, default="sample-10BT", choices=["sample-10BT", "sample-100BT", "sample-350BT", "full"])
    parser.add_argument("--use-shards", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-math", action="store_true", default=True, help="Include OpenWebMath (default: True)")
    parser.add_argument("--no-include-math", action="store_false", dest="include_math", help="Exclude OpenWebMath")
    parser.add_argument("--math-ratio", type=float, default=0.01, help="Fraction for OpenWebMath (default: 0.01 = 1%%)")
    parser.add_argument("--hf-org", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./helixlm-datasets")
    parser.add_argument("--save-local", action="store_true", default=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Samples per chunk when writing to disk")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Override CHUNK_SIZE if provided
    global CHUNK_SIZE
    if args.chunk_size:
        CHUNK_SIZE = args.chunk_size

    dataset_dict, metadata = create_dataset_chunked(
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
