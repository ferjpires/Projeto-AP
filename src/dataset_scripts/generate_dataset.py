import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.fetch_daigt import fetch_daigt
from sources.fetch_research_abstracts import fetch_research_abstracts
from sources.fetch_gsingh import fetch_gsingh
from utils import balance_classes, deduplicate, save_dataset

import pandas as pd

CLASSES_OFICIAIS = ["Human", "OpenAI", "Meta", "Google", "Anthropic"]


def generate(
    output_path="../../data/processed/dataset_combined.csv", max_per_class=2000
):
    dfs = []

    print("=" * 60)
    print("Generating combined dataset")
    print("=" * 60)

    print("\n1. DAIGT (school essays - all 5 classes)...")
    df_daigt = fetch_daigt()
    if not df_daigt.empty:
        dfs.append(df_daigt)
    else:
        print("  Skipped (no data)")

    print("\n2. Research Abstracts (scientific - Human + OpenAI)...")
    df_abstracts = fetch_research_abstracts(max_per_class=max_per_class)
    if not df_abstracts.empty:
        dfs.append(df_abstracts)
    else:
        print("  Skipped (no data)")

    print("\n3. gsingh (journalism - Human + OpenAI + Meta + Google)...")
    df_gsingh = fetch_gsingh(max_per_class=max_per_class)
    if not df_gsingh.empty:
        dfs.append(df_gsingh)
    else:
        print("  Skipped (no data)")

    if not dfs:
        print("\nERROR: No data loaded from any source!")
        return

    print("\n" + "=" * 60)
    print("Combining datasets...")
    print("=" * 60)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nBefore filtering: {len(combined)} samples")

    combined = combined[combined["Label"].isin(CLASSES_OFICIAIS)]
    print(f"After class filter: {len(combined)} samples")

    before_dedup = len(combined)
    combined = deduplicate(combined)
    after_dedup = len(combined)
    print(f"After deduplication: {after_dedup} samples")
    print(f"  Deduplication removed {before_dedup - after_dedup} rows")

    print("\nDistribution before balancing:")
    print(combined["Label"].value_counts().to_string())

    combined = balance_classes(combined)
    print(f"\nAfter balancing: {len(combined)} samples")

    df_final = save_dataset(combined, output_path, prefix="COMBINED")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"\nOutput file: {output_path}")
    print(f"Total samples: {len(df_final)}")
    print(f"\nFinal class distribution:")
    print(df_final["Label"].value_counts().to_string())


if __name__ == "__main__":
    generate()
