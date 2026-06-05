"""
Prepare Databricks Dolly-15k dataset with chat format.
Uses GPT2 tokenizer with added {im_start_str} and {im_end_str} special tokens.

The HelixLM trainer adds EOS automatically (dataset.py line 251-252), 
so we DON'T add EOS in the formatted text.

Chat format (Qwen3 style):
  {im_start_str}user
  {instruction}
  {im_end_str}
  {im_start_str}assistant
  {response}
  {im_end_str}

The EOS will be added by the trainer when processing the data.
"""
import os
import random
from datetime import datetime

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
RANDOM_SEED = 42
MAX_SEQ_LEN = 508  # Filter to < 508 tokens

# Special tokens for chat format (Qwen3 style using GPT2 base)
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'


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


def convert_to_helixlm_format(instruction, response):
    """
    Convert to HelixLM chat format using GPT2 tokenizer with im_start/im_end.

    Qwen3 style format:
      {im_start_str}user
      {instruction}
      {im_end_str}
      {im_start_str}assistant
      {response}
      {im_end_str}

    NOTE: We do NOT add EOS here because:
    1. HelixLM trainer (dataset.py lines 251-252) adds EOS automatically
    2. The format above ends with im_end which signals turn completion
    """
    formatted = f"{IM_START}user\\n{instruction}\\n{IM_END}\\n{IM_START}assistant\\n{response}\\n{IM_END}"
    return formatted


def count_tokens(text):
    """Count tokens using GPT2 tokenizer with extended vocabulary."""
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    return len(tokens)


def process_dolly_dataset(
    seed: int = RANDOM_SEED,
    val_split: float = 0.02,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """
    Load databricks-dolly-15k, apply chat format, count tokens, filter by seq len.
    
    Columns in dolly-15k:
    - instruction: user content
    - context: optional context (may be null/empty)
    - response: assistant content
    - category: task category (open_qa, closed_qa, etc.)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print("Databricks Dolly-15k - Chat Format (GPT2 + im_start/im_end)")
    print("=" * 60)
    
    # Load the dataset
    print("Loading databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"  Loaded {len(ds):,} rows")
    print(f"  Columns: {ds.column_names}")
    
    # Process each row: format conversation and count tokens
    print("\\nFormatting conversations and counting tokens...")
    
    formatted_conversations = []
    token_counts = []
    
    for i, row in enumerate(ds):
        instruction = row.get('instruction', '') or ''
        response = row.get('response', '') or ''
        
        # Strip whitespace
        instruction = instruction.strip()
        response = response.strip()
        
        # Format the conversation
        formatted = convert_to_helixlm_format(instruction, response)
        token_count = count_tokens(formatted)
        
        formatted_conversations.append(formatted)
        token_counts.append(token_count)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,}/{len(ds):,} samples...")
    
    print(f"  Done. Processed {len(ds):,} samples.")
    
    # Add new columns to the dataset
    print("\\nAdding columns to dataset...")
    ds = ds.add_column("formattedconversation", formatted_conversations)
    ds = ds.add_column("tokencount", token_counts)
    
    # Filter to < 508 tokens
    print(f"\\nFiltering to tokencount < {max_seq_len}...")
    original_len = len(ds)
    ds = ds.filter(lambda x: x['tokencount'] < max_seq_len)
    filtered_len = len(ds)
    print(f"  Filtered: {original_len:,} -> {filtered_len:,} rows ({filtered_len / original_len * 100:.1f}% retained)")
    
    if filtered_len == 0:
        raise ValueError("No samples remaining after filtering!")
    
    # Compute token statistics from filtered dataset
    filtered_token_counts = ds['tokencount']
    total_filtered_tokens = sum(filtered_token_counts)
    mean_tokens = total_filtered_tokens / filtered_len if filtered_len > 0 else 0
    
    print(f"\\nToken statistics (filtered):")
    print(f"  Mean tokens per sample: {mean_tokens:.1f}")
    print(f"  Total tokens: {total_filtered_tokens:,}")
    print(f"  Min tokens: {min(filtered_token_counts)}")
    print(f"  Max tokens: {max(filtered_token_counts)}")
    
    # Create train/val split deterministically
    print(f"\\nCreating train/val split (val={val_split*100:.0f}%)...")
    indices = np.arange(filtered_len)
    np.random.shuffle(indices)
    
    split_idx = int(filtered_len * (1 - val_split))
    train_indices = indices[:split_idx].tolist()
    val_indices = indices[split_idx:].tolist()
    
    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)
    
    dataset_dict = DatasetDict({
        "train": train_ds,
        "val": val_ds,
    })
    
    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")
    
    # Build repo name: david-thrower/databricks-dolly-{num_rows/1000}samples-{sum(tokencount)}tokens-512-seq
    num_rows_k = filtered_len // 1000
    total_tokens_k = total_filtered_tokens // 1000
    REPO_ID = f"david-thrower/databricks-dolly-{num_rows_k}K-samples-{total_tokens_k}K-tokens-512-seq"
    
    print(f"\\nRepo ID: {REPO_ID}")
    
    # Metadata
    metadata = {
        "source": "databricks/databricks-dolly-15k",
        "original_rows": original_len,
        "filtered_rows": filtered_len,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "total_tokens": total_filtered_tokens,
        "mean_tokens_per_sample": mean_tokens,
        "max_seq_len_filter": max_seq_len,
        "val_split": val_split,
        "seed": seed,
        "created": datetime.now().isoformat(),
        "format_version": "v1_dolly",
    }
    
    return dataset_dict, metadata, REPO_ID


def main():
    dataset_dict, metadata, repo_id = process_dolly_dataset(
        seed=RANDOM_SEED,
        val_split=0.02,
        max_seq_len=MAX_SEQ_LEN,
    )
    
    # Print sample formatted conversation
    print("\\n" + "=" * 60)
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
    print("\\nTokenization check:")
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
    
    print(f"\\nToken statistics:")
    print(f"  Mean: {metadata['mean_tokens_per_sample']:.0f}")
    print(f"  Total tokens: {metadata['total_tokens']:,}")
    
    # Save tokenizer config as part of dataset
    OUTPUT_DIR = repo_id.replace("/", "_")
    tokenizer_save_path = os.path.join(OUTPUT_DIR, "tokenizer")
    os.makedirs(tokenizer_save_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"\\nTokenizer saved to {tokenizer_save_path}")
    
    # Save special token info
    with open(os.path.join(OUTPUT_DIR, "special_tokens.txt"), "w") as f:
        f.write(f"IM_START: {repr(IM_START)} -> ID: {tokenizer.convert_tokens_to_ids(IM_START)}\\n")
        f.write(f"IM_END: {repr(IM_END)} -> ID: {tokenizer.convert_tokens_to_ids(IM_END)}\\n")
        f.write(f"EOS: {repr(tokenizer.eos_token)} -> ID: {tokenizer.eos_token_id}\\n")
        f.write(f"PAD: {repr(tokenizer.pad_token)} -> ID: {tokenizer.pad_token_id}\\n")
        f.write(f"\\nVocab size: {len(tokenizer)}\\n")
    
    print(f"\\nSaving dataset locally to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_dict.save_to_disk(OUTPUT_DIR)
    print("Saved to disk")
    
    print(f"\\nPushing to HuggingFace Hub: {repo_id}")
    try:
        dataset_dict.push_to_hub(repo_id, token=HF_TOKEN, private=False)
        print(f"Pushed to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"ERROR pushing to Hub: {e}")
        print(f"Dataset saved locally at: {OUTPUT_DIR}")
        raise
    
    print("\\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Examples: {metadata['filtered_rows']:,}")
    print(f"Total tokens: {metadata['total_tokens']:,}")
    print(f"Columns: {list(dataset_dict['train'].features.keys())}")
    print(f"Tokenizer: GPT2 with IM_START/IM_END")
    print(f"  IM_START id: {tokenizer.convert_tokens_to_ids(IM_START)}")
    print(f"  IM_END id: {tokenizer.convert_tokens_to_ids(IM_END)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
