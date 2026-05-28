#!/usr/bin/env python3
"""
Prepare SmolLM Corpus dataset with chat format from cosmopedia-v2 ONLY.
Target: 1,000,000 rows from HuggingFaceTB/smollm-corpus cosmopedia-v2 split
Uses GPT2 tokenizer with added <|im_start|> and <|im_end|> special tokens.

STREAMING VERSION - Memory efficient for at-scale data processing.

The HelixLM trainer adds EOS automatically (dataset.py line 251-252), 
so we DON'T add EOS in the formatted text.

Chat format (Qwen3 style):
  <|im_start|>user
  {prompt}
  <|im_end|>
  <|im_start|>assistant
  {text}
  <|im_end|>

The EOS will be added by the trainer when processing the data.
"""
import os
import random
import tempfile
import shutil
import gc
from typing import Iterator, Dict, Any, List
from datetime import datetime

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, concatenate_datasets
from transformers import AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = "smollm_corpus_cosmos_2M_gpt2_v2_streaming"
REPO_ID = "david-thrower/smollm-corpus-instruct-2M-cosmopedia-v2-gpt2-v2-streaming"
TARGET_ROWS = 1_000_000
RANDOM_SEED = 42
CHUNK_SIZE = 25000  # Process in chunks to limit memory

# Special tokens for chat format (Qwen3 style using GPT2 base)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set!")
    exit(1)
print(f"HF_TOKEN available (len={len(HF_TOKEN)})")

print("Loading GPT2 tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Add special tokens to GPT2 tokenizer
special_tokens = {
    "additional_special_tokens": [IM_START, IM_END]
}
num_added = base_tokenizer.add_special_tokens(special_tokens)

# GPT2 doesn't have a pad token by default, set it to eos_token
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

# Store the extended tokenizer for token counting
tokenizer = base_tokenizer

print(f"GPT2 tokenizer loaded")
print(f"  Original vocab size: 50257")
print(f"  Extended vocab size: {len(tokenizer)}")
print(f"  Added {num_added} new tokens")
print(f"  IM_START token id: {tokenizer.convert_tokens_to_ids(IM_START)}")
print(f"  IM_END token id: {tokenizer.convert_tokens_to_ids(IM_END)}")
print(f"  EOS token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
print(f"  PAD token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")


def convert_to_helixlm_format(prompt, text):
    """
    Convert to HelixLM chat format using GPT2 tokenizer with im_start/im_end.

    Qwen3 style format:
      <|im_start|>user
      {prompt}
      <|im_end|>
      <|im_start|>assistant
      {text}
      <|im_end|>

    NOTE: We do NOT add EOS here because:
    1. HelixLM trainer (dataset.py lines 251-252) adds EOS automatically
    2. The format above ends with im_end which signals turn completion
    """
    formatted = f"{IM_START}user\n{prompt}\n{IM_END}\n{IM_START}assistant\n{text}\n{IM_END}"
    return formatted


def count_tokens(text):
    """Count tokens using GPT2 tokenizer with extended vocabulary."""
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    return len(tokens)


def stream_cosmopedia_samples(
    num_samples: int,
    seed: int = 42,
    shuffle_buffer: int = 100000,
) -> Iterator[Dict[str, Any]]:
    """Stream samples from cosmopedia-v2 dataset."""
    print(f"    Streaming from HuggingFaceTB/smollm-corpus (cosmopedia-v2)")
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    
    count = 0
    for sample in ds:
        if count >= num_samples:
            break
        
        prompt = sample.get('prompt', '')
        text = sample.get('text', '')
        
        if prompt and text and isinstance(prompt, str) and isinstance(text, str):
            yield {
                'prompt': prompt.strip(),
                'text': text.strip(),
            }
            count += 1
    
    print(f"    Collected {count:,} samples")


def process_dataset_streaming(
    target_rows: int = TARGET_ROWS,
    seed: int = RANDOM_SEED,
    val_split: float = 0.02,
) -> tuple[DatasetDict, Dict[str, Any]]:
    """
    Memory-efficient dataset creation using streaming and chunked processing.
    
    Strategy:
    1. Stream samples from cosmopedia-v2
    2. Format and count tokens in batches
    3. Store in temporary chunked files
    4. Shuffle indices deterministically
    5. Create train/val split by selecting appropriate chunks
    """
    random.seed(seed)
    
    print("=" * 60)
    print("SmolLM Corpus - COSMOPEDIA-V2 ONLY (Streaming/GPT2 + im_start/im_end)")
    print(f"Target: {target_rows:,} rows")
    print(f"Chunk size: {CHUNK_SIZE:,} samples")
    print("=" * 60)
    
    # Collect samples in chunks to disk
    temp_dir = tempfile.mkdtemp(prefix="cosmos_chunks_")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        features = Features({
            "prompt": Value("string"),
            "text": Value("string"),
            "formattedconversation": Value("string"),
            "tokencount": Value("int64"),
        })
        
        # Stream and save in chunks
        all_chunk_files = []
        current_chunk = {"prompt": [], "text": [], "formattedconversation": [], "tokencount": []}
        chunk_num = 0
        total_samples = 0
        total_tokens = 0
        
        def save_chunk():
            nonlocal chunk_num, all_chunk_files
            if len(current_chunk["formattedconversation"]) == 0:
                return
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num:05d}.arrow")
            ds = Dataset.from_dict(current_chunk, features=features)
            ds.save_to_disk(chunk_path)
            all_chunk_files.append(chunk_path)
            print(f"    Saved chunk {chunk_num}: {len(current_chunk['formattedconversation']):,} samples")
            current_chunk["prompt"].clear()
            current_chunk["text"].clear()
            current_chunk["formattedconversation"].clear()
            current_chunk["tokencount"].clear()
            chunk_num += 1
            gc.collect()
        
        # Stream cosmopedia-v2 samples
        print("Streaming cosmopedia-v2...")
        for sample in stream_cosmopedia_samples(target_rows, seed=seed):
            # Format the conversation
            formatted = convert_to_helixlm_format(sample['prompt'], sample['text'])
            token_count = count_tokens(formatted)
            
            current_chunk["prompt"].append(sample['prompt'])
            current_chunk["text"].append(sample['text'])
            current_chunk["formattedconversation"].append(formatted)
            current_chunk["tokencount"].append(token_count)
            
            total_samples += 1
            total_tokens += token_count
            
            if len(current_chunk["formattedconversation"]) >= CHUNK_SIZE:
                save_chunk()
        
        # Save final partial chunk
        save_chunk()
        
        print(f"\nTotal samples streamed: {total_samples:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Chunks: {len(all_chunk_files)}")
        
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
        train_dataset = concatenate_datasets(train_chunks) if train_chunks else Dataset.from_dict({
            "prompt": [], "text": [], "formattedconversation": [], "tokencount": []
        }, features=features)
        val_dataset = concatenate_datasets(val_chunks) if val_chunks else Dataset.from_dict({
            "prompt": [], "text": [], "formattedconversation": [], "tokencount": []
        }, features=features)
        
        # Clean up chunks
        del train_chunks, val_chunks
        gc.collect()
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
        })
        
    finally:
        # Cleanup temp dir
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Metadata
    metadata = {
        "target_rows": target_rows,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "mean_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        "val_split": val_split,
        "seed": seed,
        "created": datetime.now().isoformat(),
        "format_version": "v2_streaming",
    }
    
    print(f"\nDataset splits:")
    print(f"  train: {len(train_dataset):,} samples")
    print(f"  val:   {len(val_dataset):,} samples")
    
    return dataset_dict, metadata


def main():
    dataset_dict, metadata = process_dataset_streaming(
        target_rows=TARGET_ROWS,
        seed=RANDOM_SEED,
        val_split=0.02,
    )
    
    # Print sample formatted conversation
    print("\n" + "=" * 60)
    print("SAMPLE FORMATTED CONVERSATION:")
    print("=" * 60)
    sample = dataset_dict["train"][0]["formattedconversation"]
    # Show the raw sample
    print(repr(sample[:1000]))
    print("-" * 60)
    # Also show it "nicely"
    print("Pretty print:")
    print(sample[:1000])
    print("=" * 60)
    
    # Show tokenization of sample
    print("\nTokenization check:")
    sample_tokens = tokenizer.encode(sample, add_special_tokens=False)
    print(f"  Total tokens in sample: {len(sample_tokens)}")
    print(f"  First 50 token IDs: {sample_tokens[:50]}")
    print(f"  Decoded back: {tokenizer.decode(sample_tokens[:50])}")
    
    # Show special token positions
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    im_start_positions = [i for i, t in enumerate(sample_tokens) if t == im_start_id]
    im_end_positions = [i for i, t in enumerate(sample_tokens) if t == im_end_id]
    print(f"  IM_START positions: {im_start_positions}")
    print(f"  IM_END positions: {im_end_positions}")
    
    print(f"\nToken statistics:")
    print(f"  Mean: {metadata['mean_tokens_per_sample']:.0f}")
    print(f"  Total tokens: {metadata['total_tokens']:,}")
    
    # Save tokenizer config as part of dataset
    tokenizer_save_path = os.path.join(OUTPUT_DIR, "tokenizer")
    os.makedirs(tokenizer_save_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"\nTokenizer saved to {tokenizer_save_path}")
    
    # Save special token info
    with open(os.path.join(OUTPUT_DIR, "special_tokens.txt"), "w") as f:
        f.write(f"IM_START: {repr(IM_START)} -> ID: {tokenizer.convert_tokens_to_ids(IM_START)}\n")
        f.write(f"IM_END: {repr(IM_END)} -> ID: {tokenizer.convert_tokens_to_ids(IM_END)}\n")
        f.write(f"EOS: {repr(tokenizer.eos_token)} -> ID: {tokenizer.eos_token_id}\n")
        f.write(f"PAD: {repr(tokenizer.pad_token)} -> ID: {tokenizer.pad_token_id}\n")
        f.write(f"\nVocab size: {len(tokenizer)}\n")
    
    print(f"\nSaving dataset locally to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_dict.save_to_disk(OUTPUT_DIR)
    print("Saved to disk")
    
    print(f"\nPushing to HuggingFace Hub: {REPO_ID}")
    try:
        dataset_dict.push_to_hub(REPO_ID, token=HF_TOKEN, private=False)
        print(f"Pushed to: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"ERROR pushing to Hub: {e}")
        print(f"Dataset saved locally at: {OUTPUT_DIR}")
        raise
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Examples: {metadata['total_samples']:,}")
    print(f"Total tokens: {metadata['total_tokens']:,}")
    print(f"Columns: {list(dataset_dict['train'].features.keys())}")
    print(f"Tokenizer: GPT2 with IM_START/IM_END")
    print(f"  IM_START id: {tokenizer.convert_tokens_to_ids(IM_START)}")
    print(f"  IM_END id: {tokenizer.convert_tokens_to_ids(IM_END)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
