#!/usr/bin/env python3
"""
Create 3B token dataset from:
- 99%: HuggingFaceTB/smollm-corpus (fineweb-edu-dedup, English only)
- 1%: open-web-math/open-web-math (train split)

Uses HuggingFace Datasets streaming API for memory-efficient shuffle/split.

Run as: nohup python create_3b_token_dataset_hfapi.py > dataset_creation.log 2>&1 &
"""

import os
import json
import gc
import time
from pathlib import Path
from typing import Iterator, Dict, List
from dataclasses import dataclass, asdict

import datasets
from datasets import (
    load_dataset, 
    IterableDataset, 
    Dataset, 
    DatasetDict, 
    Features, 
    Value,
    concatenate_datasets,
)
from transformers import AutoTokenizer
from huggingface_hub import HfApi

# Configuration
TARGET_TOKENS = 3_000_000_000  # 3B tokens
TOKENIZER_NAME = "gpt2"
SHARD_SIZE = 100_000  # Examples per shard for temp storage
VAL_RATIO = 0.02  # 2% validation
RANDOM_SEED = 42
SHUFFLE_BUFFER_SIZE = 100_000  # HF shuffle buffer (memory-efficient)

# Output paths
OUTPUT_DIR = Path("/home/ubuntu/ml-intern/dataset_3b_output")
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# Hub config
HF_REPO_NAME = "david-thrower/helixlm87M-3Btoken-pretrain-dataset-v1"

# Dataset sources
DATASET_CONFIGS = {
    "fineweb_edu": {
        "path": "HuggingFaceTB/smollm-corpus",
        "subset": "fineweb-edu-dedup",
        "split": "train",
        "column": "text",
        "filter_fn": lambda ex: ex.get("language", "") == "en" if isinstance(ex.get("language"), str) else True,
        "target_ratio": 0.99,
    },
    "open_web_math": {
        "path": "open-web-math/open-web-math",
        "subset": None,
        "split": "train",
        "column": "text",
        "filter_fn": None,
        "target_ratio": 0.01,
    },
}


@dataclass
class ProgressTracker:
    total_tokens: int = 0
    tokens_per_source: Dict[str, int] = None
    examples_per_source: Dict[str, int] = None
    examples_collected: int = 0

    def __post_init__(self):
        if self.tokens_per_source is None:
            self.tokens_per_source = {}
        if self.examples_per_source is None:
            self.examples_per_source = {}

    def add(self, source: str, tokens: int, examples: int = 1):
        self.total_tokens += tokens
        self.tokens_per_source[source] = self.tokens_per_source.get(source, 0) + tokens
        self.examples_per_source[source] = self.examples_per_source.get(source, 0) + examples
        self.examples_collected += examples

    def to_dict(self):
        return asdict(self)


def get_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def estimate_tokens(text: str, tokenizer) -> int:
    return max(1, len(text) // 4)  # ~4 chars per token


def stream_filtered_dataset(config: Dict, tokenizer) -> Iterator[Dict]:
    """Stream a dataset with optional filtering."""
    path = config["path"]
    subset = config.get("subset")
    split = config["split"]
    text_col = config["column"]
    filter_fn = config.get("filter_fn")
    source_name = config.get("name", path.split("/")[-1])

    print(f"\nStreaming from: {path}" + (f" [{subset}]" if subset else ""))
    print(f"  Split: {split}, Text column: {text_col}")

    # Load as streaming IterableDataset
    try:
        if subset:
            ds = load_dataset(path, subset, split=split, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(path, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return

    count = 0
    for example in ds:
        # Apply filter if provided
        if filter_fn and not filter_fn(example):
            continue

        text = example.get(text_col, "")
        if not text or len(text) < 50:
            continue

        yield {
            "text": text,
            "source": source_name,
            "token_count": estimate_tokens(text, tokenizer),
        }

        count += 1
        if count % 10000 == 0:
            print(f"  ... streamed {count:,} examples from {source_name}")


def create_mixed_streaming_dataset(
    configs: Dict,
    target_tokens: int,
    tokenizer,
) -> Iterator[Dict]:
    """
    Create a streaming iterator that mixes datasets according to ratios.
    This yields dicts to be consumed by IterableDataset.from_generator().
    """
    target_tokens_per_source = {
        name: int(target_tokens * cfg["target_ratio"])
        for name, cfg in configs.items()
    }

    print("\n" + "=" * 60)
    print("DATASET MIXING PLAN")
    print("=" * 60)
    for name, tokens in target_tokens_per_source.items():
        print(f"  {name}: {tokens:,} target tokens ({configs[name]['target_ratio']*100:.1f}%)")
    print("=" * 60)

    tokens_collected = {name: 0 for name in configs}

    for source_name, config in configs.items():
        target = target_tokens_per_source[source_name]
        config["name"] = source_name

        print(f"\nCollecting from {source_name} (target: {target:,} tokens)...")
        
        for example in stream_filtered_dataset(config, tokenizer):
            yield example

            tokens_collected[source_name] += example.get("token_count", 0)

            if tokens_collected[source_name] >= target:
                print(f"✓ Reached target for {source_name}: {tokens_collected[source_name]:,} tokens")
                break

            total = sum(tokens_collected.values())
            if total >= target_tokens:
                break

        total = sum(tokens_collected.values())
        if total >= target_tokens:
            break

    print(f"\nCollection complete. Total tokens per source:")
    for name, tok in tokens_collected.items():
        print(f"  {name}: {tok:,} tokens")


def write_shards_for_checkpoint(
    iterator: Iterator[Dict],
    output_dir: Path,
    tracker: ProgressTracker,
    target_tokens: int,
) -> List[Path]:
    """
    Write streaming data to JSONL shards for checkpointing.
    These are intermediate files that will be re-loaded as IterableDataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []

    current_shard = []
    current_shard_tokens = 0
    shard_idx = 0
    total_examples = 0

    print(f"\nWriting checkpoint shards to: {output_dir}")
    print(f"Target tokens: {target_tokens:,}")
    
    start_time = time.time()

    for example in iterator:
        current_shard.append({"text": example["text"], "source": example["source"]})
        current_shard_tokens += example.get("token_count", 0)
        tracker.add(example["source"], example.get("token_count", 0))
        total_examples += 1

        if len(current_shard) >= SHARD_SIZE:
            shard_path = output_dir / f"shard_{shard_idx:05d}.jsonl"
            
            # Write shard
            with open(shard_path, "w", encoding="utf-8") as f:
                for ex in current_shard:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
            shard_paths.append(shard_path)
            
            elapsed = time.time() - start_time
            rate = tracker.total_tokens / elapsed if elapsed > 0 else 0
            print(f"Shard {shard_idx:05d}: {len(current_shard):,} examples, "
                  f"Total: {tracker.total_tokens:,} tokens ({rate/1e6:.2f}M tok/s)")

            current_shard = []
            current_shard_tokens = 0
            shard_idx += 1

            if tracker.total_tokens >= target_tokens:
                print(f"\n✓ Reached target of {target_tokens:,} tokens!")
                break

            if shard_idx % 10 == 0:
                save_progress(tracker)
                gc.collect()

    # Write final partial shard
    if current_shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for ex in current_shard:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        shard_paths.append(shard_path)
        print(f"Final shard {shard_idx:05d}: {len(current_shard):,} examples")

    tracker.examples_collected = total_examples
    return shard_paths


def save_progress(tracker: ProgressTracker):
    """Save progress to metadata file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump({
            "progress": tracker.to_dict(),
            "timestamp": time.time(),
        }, f, indent=2)
    print(f"  Saved progress: {tracker.total_tokens:,} tokens")


def create_split_datasets_from_shards(shard_dir: Path, tracker: ProgressTracker):
    """
    Load shards as IterableDataset, shuffle, and split using HF Datasets API.
    
    Key: We convert to map-style Dataset AFTER sharding to enable train_test_split,
    but do it in batches to avoid OOM.
    """
    print("\n" + "=" * 60)
    print("STEP: CONVERTING TO DATASET WITH TRAIN/VAL SPLIT")
    print("=" * 60)
    
    total_examples = tracker.examples_collected
    val_size = int(total_examples * VAL_RATIO)
    train_size = total_examples - val_size
    
    print(f"Total examples: {total_examples:,}")
    print(f"Train: {train_size:,} ({(1-VAL_RATIO)*100:.0f}%)")
    print(f"Val: {val_size:,} ({VAL_RATIO*100:.0f}%)")
    
    # Load shards as IterableDataset (streaming, OOM-safe)
    print("\nLoading sharded data as streaming dataset...")
    
    # Use HF's load_dataset with streaming
    shard_files = [str(p) for p in sorted(shard_dir.glob("shard_*.jsonl"))]
    print(f"Found {len(shard_files)} shard files")
    
    # Load as streaming IterableDataset
    streaming_ds = load_dataset(
        "json", 
        data_files=shard_files, 
        streaming=True,
        split="train"
    )
    
    # Shuffle using HF's buffer-based shuffle (memory-efficient)
    print(f"Shuffling with buffer_size={SHUFFLE_BUFFER_SIZE:,}...")
    shuffled = streaming_ds.shuffle(seed=RANDOM_SEED, buffer_size=SHUFFLE_BUFFER_SIZE)
    
    # Split using take/skip (streaming-compatible)
    print("Splitting into train/val...")
    val_ds_iter = shuffled.take(val_size)
    train_ds_iter = shuffled.skip(val_size)
    
    # To enable saving, we need to materialize, but we do it in manageable chunks
    # For very large datasets, use to_parquet directly without converting to Dataset
    
    return train_ds_iter, val_ds_iter, train_size, val_size


def materialize_and_save(
    train_iter, 
    val_iter, 
    train_size: int, 
    val_size: int,
    output_dir: Path
) -> DatasetDict:
    """
    Materialize streaming datasets and save to disk.
    
    For 3B tokens ~ 2.5M examples, we can materialize if we have enough RAM.
    But we'll do it carefully with progress tracking.
    """
    print("\n" + "=" * 60)
    print("STEP: MATERIALIZING AND SAVING")
    print("=" * 60)
    
    train_dir = output_dir / "train_temp"
    val_dir = output_dir / "val_temp"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Materialize val (smaller) - can do directly
    print(f"\nMaterializing val dataset ({val_size:,} examples)...")
    val_list = []
    for i, ex in enumerate(val_iter):
        val_list.append(ex)
        if (i + 1) % 10000 == 0:
            print(f"  Val: {i+1:,}/{val_size:,}")
    print(f"✓ Val materialized: {len(val_list):,} examples")
    
    # Materialize train in chunks
    print(f"\nMaterializing train dataset ({train_size:,} examples)...")
    train_shards = []
    chunk = []
    chunk_idx = 0
    
    for i, ex in enumerate(train_iter):
        chunk.append(ex)
        
        if len(chunk) >= 50000:  # 50k example chunks
            # Save chunk to disk
            chunk_path = train_dir / f"chunk_{chunk_idx:05d}.jsonl"
            with open(chunk_path, "w", encoding="utf-8") as f:
                for item in chunk:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            train_shards.append(chunk_path)
            print(f"  Saved chunk {chunk_idx}: {len(chunk):,} examples ({i+1:,}/{train_size:,})")
            chunk = []
            chunk_idx += 1
            gc.collect()
    
    # Save final chunk
    if chunk:
        chunk_path = train_dir / f"chunk_{chunk_idx:05d}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        train_shards.append(chunk_path)
        print(f"  Saved final chunk {chunk_idx}: {len(chunk):,} examples")
    
    # Now load as HF Datasets
    print("\nLoading train from disk as Dataset...")
    train_files = [str(p) for p in sorted(train_dir.glob("chunk_*.jsonl"))]
    train_ds = load_dataset("json", data_files=train_files, split="train")
    print(f"✓ Train dataset loaded: {len(train_ds):,} examples")
    
    # Create val Dataset from list
    print("Creating val Dataset...")
    features = Features({
        "text": Value("string"),
        "source": Value("string"),
    })
    val_ds = Dataset.from_list(val_list, features=features)
    print(f"✓ Val dataset loaded: {len(val_ds):,} examples")
    
    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})
    
    # Cleanup temp files
    print("\nCleaning up temporary chunk files...")
    for f in train_dir.glob("chunk_*.jsonl"):
        f.unlink()
    train_dir.rmdir()
    val_dir.rmdir()
    
    return dataset_dict


def push_to_hub(dataset_dict: DatasetDict, repo_name: str):
    """Push dataset to HuggingFace Hub."""
    print("\n" + "=" * 60)
    print("PUSHING TO HUGGINGFACE HUB")
    print("=" * 60)
    print(f"Repository: {repo_name}")

    try:
        dataset_dict.push_to_hub(
            repo_name,
            private=False,
            commit_message="3B token mixed dataset: 99% Fineweb-Edu + 1% Open-Web-Math (HF API version)",
        )
        print("\n✅ Successfully pushed to Hub!")
        return True
    except Exception as e:
        print(f"\n❌ Failed to push to Hub: {e}")
        return False


def main():
    start_time = time.time()

    print("=" * 60)
    print("3B TOKEN DATASET CREATION (HF DATASETS API)")
    print("=" * 60)
    print(f"Target: {TARGET_TOKENS:,} tokens")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Repository: {HF_REPO_NAME}")
    print(f"Shuffle buffer: {SHUFFLE_BUFFER_SIZE:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_shards_dir = OUTPUT_DIR / "tmp_shards"
    tokenizer = get_tokenizer()
    tracker = ProgressTracker()

    # Phase 1: Stream and shard (if not already done)
    if not any(tmp_shards_dir.glob("shard_*.jsonl")):
        print("\n" + "=" * 60)
        print("PHASE 1: STREAMING AND SHARDING")
        print("=" * 60)

        # Create streaming iterator
        mixed_iterator = create_mixed_streaming_dataset(
            DATASET_CONFIGS, 
            TARGET_TOKENS, 
            tokenizer
        )
        
        # Write to shards for checkpointing
        shard_paths = write_shards_for_checkpoint(
            mixed_iterator,
            tmp_shards_dir,
            tracker,
            TARGET_TOKENS,
        )
        save_progress(tracker)
        print(f"\n✓ Wrote {len(shard_paths)} shards")
    else:
        print(f"\n✓ Found {len(list(tmp_shards_dir.glob('shard_*.jsonl')))} existing shards")
        # Load metadata
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                meta = json.load(f)
                tracker = ProgressTracker(**meta["progress"])
            print(f"  Loaded progress: {tracker.total_tokens:,} tokens, {tracker.examples_collected:,} examples")

    # Phase 2: Shuffle and split
    train_iter, val_iter, train_size, val_size = create_split_datasets_from_shards(
        tmp_shards_dir, 
        tracker
    )

    # Phase 3: Materialize and save
    dataset_dict = materialize_and_save(
        train_iter, 
        val_iter, 
        train_size, 
        val_size,
        OUTPUT_DIR
    )

    # Phase 4: Push to Hub
    success = push_to_hub(dataset_dict, HF_REPO_NAME)

    # Phase 5: Save locally
    final_dir = OUTPUT_DIR / "final_dataset"
    print("\n" + "=" * 60)
    print("SAVING TO DISK")
    print("=" * 60)
    print(f"Saving to: {final_dir}")
    dataset_dict.save_to_disk(final_dir)
    print("✓ Saved successfully")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"Elapsed time: {elapsed/3600:.2f} hours")
    print(f"Train examples: {len(dataset_dict['train']):,}")
    print(f"Validation examples: {len(dataset_dict['validation']):,}")
    print(f"Estimated tokens: {tracker.total_tokens:,}")
    print(f"Local output: {final_dir}")
    if success:
        print(f"Hub URL: https://huggingface.co/datasets/{HF_REPO_NAME}")


if __name__ == "__main__":
    main()
