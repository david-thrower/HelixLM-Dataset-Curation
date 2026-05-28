#!/usr/bin/env python3
"""
Prepare SmolLM Corpus dataset with chat format from cosmopedia-v2 ONLY.
Target: 2,000,000 rows from HuggingFaceTB/smollm-corpus cosmopedia-v2 split
Uses GPT2 tokenizer with added <|im_start|> and <|im_end|> special tokens.

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
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = "smollm_corpus_cosmos_2M_gpt2_v2"
REPO_ID = "david-thrower/smollm-corpus-instruct-2M-cosmopedia-v2-gpt2-v2"
TARGET_ROWS = 2_000_000
RANDOM_SEED = 42

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
    formatted = f"{IM_START}user\n{prompt.strip()}\n{IM_END}\n{IM_START}assistant\n{text.strip()}\n{IM_END}"
    return formatted


def count_tokens(text):
    """Count tokens using GPT2 tokenizer with extended vocabulary."""
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    return len(tokens)


def main():
    print("=" * 60)
    print("SmolLM Corpus - COSMOPEDIA-V2 ONLY (GPT2 + im_start/im_end)")
    print(f"Target: {TARGET_ROWS:,} rows")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    print("\nLoading cosmopedia-v2 split...")
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train")
    total_available = len(ds)
    print(f"  Total available: {total_available:,}")

    if TARGET_ROWS >= total_available:
        print(f"  Taking all {total_available:,} rows")
        selected = ds
    else:
        print(f"  Sampling {TARGET_ROWS:,} rows randomly")
        ds = ds.shuffle(seed=RANDOM_SEED)
        selected = ds.select(range(TARGET_ROWS))

    print(f"  Selected: {len(selected):,} rows")

    print("\nConverting to DataFrame...")
    df = selected.to_pandas()
    print(f"DataFrame shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    if 'prompt' not in df.columns or 'text' not in df.columns:
        print("ERROR: Required columns 'prompt' and 'text' not found!")
        exit(1)

    print("\nConverting to HelixLM chat format...")
    df['formattedconversation'] = df.apply(
        lambda row: convert_to_helixlm_format(row['prompt'], row['text']),
        axis=1
    )
    print("Format conversion complete")

    print("\n" + "=" * 60)
    print("SAMPLE FORMATTED CONVERSATION:")
    print("=" * 60)
    sample = df['formattedconversation'].iloc[0]
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

    print("\nCounting tokens...")
    df['tokencount'] = df['formattedconversation'].apply(count_tokens)

    print(f"\nToken statistics:")
    print(f"  Mean: {df['tokencount'].mean():.0f}")
    print(f"  Median: {df['tokencount'].median():.0f}")
    print(f"  Min: {df['tokencount'].min()}")
    print(f"  Max: {df['tokencount'].max()}")
    print(f"  Total tokens: {df['tokencount'].sum():,}")

    print(f"\nFinal columns: {df.columns.tolist()}")

    print("\nCreating HuggingFace Dataset...")
    stock_ds = Dataset.from_pandas(df)
    print(f"Dataset created with {len(stock_ds)} examples")

    # Save tokenizer config as part of dataset
    tokenizer_save_path = os.path.join(OUTPUT_DIR, "tokenizer")
    os.makedirs(tokenizer_save_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer saved to {tokenizer_save_path}")

    # Save special token info
    with open(os.path.join(OUTPUT_DIR, "special_tokens.txt"), "w") as f:
        f.write(f"IM_START: {repr(IM_START)} -> ID: {tokenizer.convert_tokens_to_ids(IM_START)}\n")
        f.write(f"IM_END: {repr(IM_END)} -> ID: {tokenizer.convert_tokens_to_ids(IM_END)}\n")
        f.write(f"EOS: {repr(tokenizer.eos_token)} -> ID: {tokenizer.eos_token_id}\n")
        f.write(f"PAD: {repr(tokenizer.pad_token)} -> ID: {tokenizer.pad_token_id}\n")
        f.write(f"\nVocab size: {len(tokenizer)}\n")

    print(f"\nSaving dataset to disk: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stock_ds.save_to_disk(OUTPUT_DIR)
    print("Saved to disk")

    print(f"\nPushing to HuggingFace Hub: {REPO_ID}")
    try:
        stock_ds.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"Pushed to: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"ERROR pushing to Hub: {e}")
        print(f"Dataset saved locally at: {OUTPUT_DIR}")
        raise

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Examples: {len(stock_ds):,}")
    print(f"Total tokens: {df['tokencount'].sum():,}")
    print(f"Columns preserved: {df.columns.tolist()}")
    print(f"Tokenizer: GPT2 with IM_START/IM_END")
    print(f"  IM_START id: {tokenizer.convert_tokens_to_ids(IM_START)}")
    print(f"  IM_END id: {tokenizer.convert_tokens_to_ids(IM_END)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
