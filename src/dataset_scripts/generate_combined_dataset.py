import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DATA_RAW_GENERATED, DATA_PROCESSED,
    load_generated_csv, deduplicate, balance_classes, save_dataset,
)

CLASSES = ["Human", "OpenAI", "Meta", "Google", "Anthropic"]
DEFAULT_OUTPUT = os.path.join(DATA_PROCESSED, "dataset_combined.csv")


def word_count_filter(df, min_words=81, max_words=120):
    wc = df["Text"].apply(lambda x: len(str(x).split()))
    before = len(df)
    df = df[(wc >= min_words) & (wc <= max_words)].copy()
    removed = before - len(df)
    if removed:
        print(f"  Word filter ({min_words}-{max_words}): removed {removed}, kept {len(df)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Combine generated CSVs into training dataset")
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="Max samples per class after balancing")
    parser.add_argument("--min-words", type=int, default=90, help="Min words per text")
    parser.add_argument("--max-words", type=int, default=130, help="Max words per text")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Combining generated datasets")
    print("=" * 60)

    csv_files = sorted(glob.glob(os.path.join(DATA_RAW_GENERATED, "*.csv")))

    if not csv_files:
        print(f"ERROR: No CSV files found in {DATA_RAW_GENERATED}")
        sys.exit(1)

    dfs = []
    for path in csv_files:
        name = os.path.basename(path)
        try:
            df = load_generated_csv(path)
            if df.empty:
                print(f"  SKIP: {name} — empty")
                continue

            df = df[df["Label"].isin(CLASSES)].copy()
            if df.empty:
                print(f"  SKIP: {name} — no known labels")
                continue

            labels = df["Label"].value_counts().to_dict()
            avg_words = df["Text"].apply(lambda x: len(str(x).split())).mean()
            print(f"  {name}: {len(df)} rows, avg {avg_words:.0f} words — {labels}")
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    if not dfs:
        print("\nERROR: No data loaded from any source!")
        sys.exit(1)

    import pandas as pd
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} samples")
    print(combined["Label"].value_counts().sort_index().to_string())

    combined = word_count_filter(combined, args.min_words, args.max_words)

    before = len(combined)
    combined = deduplicate(combined)
    removed = before - len(combined)
    if removed:
        print(f"\nRemoved {removed} duplicates → {len(combined)} remaining")

    combined = balance_classes(
        combined, max_per_class=args.max_per_class, random_state=args.seed
    )
    print(f"\nAfter balancing ({args.max_per_class}/class):")
    print(combined["Label"].value_counts().sort_index().to_string())

    save_dataset(combined, args.output, prefix="COMBINED")

    print(f"\nTotal: {len(combined)} samples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
