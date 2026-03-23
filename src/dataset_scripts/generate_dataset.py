import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.fetch_daigt import fetch_daigt
from sources.fetch_research_abstracts import fetch_research_abstracts
from sources.fetch_gsingh import fetch_gsingh
from sources.fetch_ai_models import (
    fetch_meta,
    fetch_google,
    fetch_openai_from_csv,
    fetch_anthropic_from_csv,
)
from utils import balance_classes, deduplicate, save_dataset

import pandas as pd

CLASSES = ["Human", "OpenAI", "Meta", "Google", "Anthropic"]

MANUAL_OPENAI_CSV    = "../../data/raw/manual_openai.csv"
MANUAL_ANTHROPIC_CSV = "../../data/raw/manual_anthropic.csv"
GENERATED_META_CSV   = "../../data/raw/generated_meta.csv"
GENERATED_GOOGLE_CSV = "../../data/raw/generated_google.csv"

def generate(
    output_path: str = "../../data/processed/dataset_combined.csv",
    daigt_max_per_class: int = 150,
    existing_max_per_class: int = 400,
    max_per_class: int = 500,
    api_topics: list = None,
    api_variants: int = 3,
    api_delay: float = 2.0,
    skip_api: bool = False,
):
    dfs = []

    print("=" * 60)
    print("Generating combined dataset")
    print("=" * 60)

    print(f"\n[1/6] DAIGT (school essays, max {daigt_max_per_class}/class)...")
    df_daigt = fetch_daigt(max_per_class=daigt_max_per_class)
    if not df_daigt.empty:
        dfs.append(df_daigt)

    print(f"\n[2/6] Research Abstracts (scientific, max {existing_max_per_class}/class)...")
    df_abstracts = fetch_research_abstracts(max_per_class=existing_max_per_class)
    if not df_abstracts.empty:
        dfs.append(df_abstracts)

    print(f"\n[3/6] gsingh (journalism, max {existing_max_per_class}/class)...")
    df_gsingh = fetch_gsingh(max_per_class=existing_max_per_class)
    if not df_gsingh.empty:
        dfs.append(df_gsingh)

    if not skip_api:
        print(f"\n[4/6] Generating scientific texts via API...")

        print("\n  → Meta (Groq / llama-3.3-70b)...")
        try:
            df_meta_api = fetch_meta(
                topics=api_topics,
                variants=api_variants,
                delay=api_delay,
            )
            if not df_meta_api.empty:
                df_meta_api.to_csv(GENERATED_META_CSV, index=False)
                print(f"  Saved to {GENERATED_META_CSV}")
                dfs.append(df_meta_api)
        except Exception as e:
            print(f"  Meta API FAILED: {e}")

        print("\n  → Google (Gemini API)...")
        try:
            df_google_api = fetch_google(
                topics=api_topics,
                variants=api_variants,
                delay=api_delay,
            )
            if not df_google_api.empty:
                df_google_api.to_csv(GENERATED_GOOGLE_CSV, index=False)
                print(f"  Saved to {GENERATED_GOOGLE_CSV}")
                dfs.append(df_google_api)
        except Exception as e:
            print(f"  Google API FAILED: {e}")

    else:
        print("\n[4/6] Loading pre-generated scientific texts from CSV...")
        for path, label in [(GENERATED_META_CSV, "Meta"), (GENERATED_GOOGLE_CSV, "Google")]:
            if os.path.exists(path):
                df_cached = pd.read_csv(path)
                dfs.append(df_cached)
                print(f"  Loaded {len(df_cached)} {label} samples from {path}")
            else:
                print(f"  WARNING: {path} not found — run without --skip-api first")

    print(f"\n[5/6] Manual OpenAI texts ({MANUAL_OPENAI_CSV})...")
    df_openai_manual = fetch_openai_from_csv(MANUAL_OPENAI_CSV)
    if not df_openai_manual.empty:
        dfs.append(df_openai_manual)

    print(f"\n[6/6] Manual Anthropic texts ({MANUAL_ANTHROPIC_CSV})...")
    df_anthropic_manual = fetch_anthropic_from_csv(MANUAL_ANTHROPIC_CSV)
    if not df_anthropic_manual.empty:
        dfs.append(df_anthropic_manual)

    if not dfs:
        print("\nERROR: No data loaded from any source!")
        return None

    print("\n" + "=" * 60)
    print("Combining and cleaning...")
    print("=" * 60)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["Label"].isin(CLASSES)]
    print(f"\nAfter class filter: {len(combined)} samples")
    print("Raw distribution:")
    print(combined["Label"].value_counts().to_string())

    before = len(combined)
    combined = deduplicate(combined)
    print(f"\nAfter deduplication: {len(combined)} ({before - len(combined)} removed)")

    combined = balance_classes(combined, max_per_class=max_per_class)
    print(f"\nAfter balancing: {len(combined)} samples")
    print(combined["Label"].value_counts().to_string())

    df_final = save_dataset(combined, output_path, prefix="COMBINED")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total: {len(df_final)} samples, {len(CLASSES)} classes")

    return df_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--daigt-max", type=int, default=150)
    parser.add_argument("--max-per-class", type=int, default=500)
    args = parser.parse_args()

    generate(
        skip_api=args.skip_api,
        api_variants=args.variants,
        daigt_max_per_class=args.daigt_max,
        max_per_class=args.max_per_class,
    )