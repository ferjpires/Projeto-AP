import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from utils import truncate_text

LABEL_MAP = {
    "Human_story": "Human",
    "GPT_4-o": "OpenAI",
    "llama-8B": "Meta",
    "gemma-2-9b": "Google",
}


def fetch_gsingh(max_per_class=2000) -> pd.DataFrame:
    print("Loading gsingh dataset (NYT + LLMs)...")

    try:
        from datasets import load_dataset

        ds = load_dataset("gsingh1-py/train", split="train")
    except Exception as e:
        print(f"  WARNING: Could not load gsingh dataset: {e}")
        return pd.DataFrame(columns=["Text", "Label"])

    df = ds.to_pandas()
    print(f"  Available columns: {df.columns.tolist()}")

    dfs = []
    for col, label in LABEL_MAP.items():
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found, skipping")
            continue

        # Handle NaN per column independently
        subset = df[[col]].dropna().rename(columns={col: "Text"})
        subset = subset[subset["Text"].str.strip() != ""]
        subset["Label"] = label
        subset["Text"] = subset["Text"].apply(lambda x: truncate_text(str(x), 120))

        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=42)

        dfs.append(subset)
        print(f"  {label}: {len(subset)} samples")

    if not dfs:
        print("  WARNING: No data loaded")
        return pd.DataFrame(columns=["Text", "Label"])

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(result)} samples")
    return result
