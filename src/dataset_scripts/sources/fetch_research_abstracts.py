import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from utils import truncate_text


def fetch_research_abstracts(max_per_class=2000) -> pd.DataFrame:
    print("Loading Research Abstracts dataset (scientific domain)...")

    try:
        from datasets import load_dataset

        ds = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts", split="train")
    except Exception as e:
        print(f"  WARNING: Could not load Research Abstracts dataset: {e}")
        return pd.DataFrame(columns=["Text", "Label"])

    df = ds.to_pandas()
    print(f"  Columns: {df.columns.tolist()}")

    dfs = []
    for col, label in [("real_abstract", "Human"), ("generated_abstract", "OpenAI")]:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found")
            continue

        subset = df[[col]].dropna().rename(columns={col: "Text"})
        subset["Label"] = label
        subset["Text"] = subset["Text"].apply(lambda x: truncate_text(str(x), 120))

        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=42)

        dfs.append(subset)
        print(f"  {label}: {len(subset)} samples")

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(result)} samples")

    return result
