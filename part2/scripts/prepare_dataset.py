"""
Prepare the MIQR-CC dataset using the official train/val/test split.

The official split is expected at:
    data/official_split/<split>/<class>/*.png

This script enriches data/metadata.csv with the official split information and
then rebuilds data/processed/ from that metadata.
"""

from pathlib import Path
import shutil
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "metadata.csv"
RAW_DIR = DATA_DIR / "raw"
OFFICIAL_SPLIT_DIR = DATA_DIR / "official_split"
OUTPUT_DIR = DATA_DIR / "processed"

CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]
SPLITS = ["train", "val", "test"]

LABEL_MAP = {
    "Biliary Leaks": "Biliary_Leaks",
    "Lithiasis": "Lithiasis",
    "Normal": "Normal",
    "Benign Stricture": "Stricture",
    "Malignant Stricture": "Stricture",
}

EXPECTED_COUNTS = {
    ("train", "Biliary_Leaks"): 110,
    ("train", "Lithiasis"): 505,
    ("train", "Normal"): 197,
    ("train", "Stricture"): 255,
    ("val", "Biliary_Leaks"): 24,
    ("val", "Lithiasis"): 98,
    ("val", "Normal"): 59,
    ("val", "Stricture"): 53,
    ("test", "Biliary_Leaks"): 17,
    ("test", "Lithiasis"): 123,
    ("test", "Normal"): 43,
    ("test", "Stricture"): 84,
}


def main():
    if not METADATA_CSV.exists():
        fail(f"Metadata not found: {METADATA_CSV}")
    if not OFFICIAL_SPLIT_DIR.exists():
        fail(f"Official split directory not found: {OFFICIAL_SPLIT_DIR}")

    df = pd.read_csv(METADATA_CSV)
    print(f"Metadata rows: {len(df)}")

    df = enrich_metadata_with_official_split(df)
    df.to_csv(METADATA_CSV, index=False)
    print(f"Metadata updated with official split columns: {METADATA_CSV}")

    official_df = df[df["official_split"].isin(SPLITS)].copy()
    validate_expected_counts(official_df, source="metadata")
    validate_sources_exist(official_df)

    rebuild_processed_dataset(official_df)
    validate_processed_counts()

    print("\nDone. data/processed now uses the official split.")


def enrich_metadata_with_official_split(df: pd.DataFrame) -> pd.DataFrame:
    required = {"processed_image_path", "Label", "Keep"}
    missing = required - set(df.columns)
    if missing:
        fail(f"Missing required metadata columns: {sorted(missing)}")

    df = df.copy()
    metadata_names = {Path(p).name for p in df["processed_image_path"]}
    official_map = build_official_split_map(metadata_names)

    normalized_names = df["processed_image_path"].map(lambda p: Path(p).name)
    df["class_name"] = df["Label"].map(LABEL_MAP)
    df["official_split"] = normalized_names.map(
        lambda name: official_map.get(name, {}).get("split", "")
    )
    df["official_class_name"] = normalized_names.map(
        lambda name: official_map.get(name, {}).get("class_name", "")
    )
    df["official_filename"] = normalized_names.map(
        lambda name: official_map.get(name, {}).get("official_filename", "")
    )
    df["official_source_path"] = normalized_names.map(
        lambda name: official_map.get(name, {}).get("official_source_path", "")
    )
    df["is_official_split"] = df["official_split"].isin(SPLITS)

    official_rows = df[df["is_official_split"]]
    mismatched = official_rows[
        official_rows["class_name"] != official_rows["official_class_name"]
    ]
    if not mismatched.empty:
        cols = ["processed_image_path", "Label", "class_name", "official_class_name"]
        print(mismatched[cols].head(20).to_string(index=False))
        fail("Official split class does not match metadata label.")

    if len(official_rows) != sum(EXPECTED_COUNTS.values()):
        fail(
            "Official split row count mismatch: "
            f"{len(official_rows)} != {sum(EXPECTED_COUNTS.values())}"
        )

    return df


def build_official_split_map(metadata_names: set[str]) -> dict[str, dict[str, str]]:
    official_map = {}

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            class_dir = OFFICIAL_SPLIT_DIR / split / class_name
            if not class_dir.exists():
                fail(f"Missing official split class directory: {class_dir}")

            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue

                normalized = normalize_official_filename(path.name, metadata_names)
                if normalized not in metadata_names:
                    fail(f"Official file not present in metadata.csv: {path.name}")
                if normalized in official_map:
                    fail(f"Duplicate official split entry for: {normalized}")

                official_map[normalized] = {
                    "split": split,
                    "class_name": class_name,
                    "official_filename": path.name,
                    "official_source_path": str(path.relative_to(PROJECT_ROOT)),
                }

    return official_map


def normalize_official_filename(filename: str, metadata_names: set[str]) -> str:
    if filename in metadata_names:
        return filename

    if "_" in filename:
        stripped = filename.split("_", 1)[1]
        if stripped in metadata_names:
            return stripped

    return filename


def validate_expected_counts(df: pd.DataFrame, source: str) -> None:
    counts = df.groupby(["official_split", "official_class_name"]).size().to_dict()
    errors = []

    for key, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(key, 0))
        if actual != expected:
            errors.append((key, expected, actual))

    if errors:
        for (split, class_name), expected, actual in errors:
            print(f"{source}: {split}/{class_name}: expected {expected}, got {actual}")
        fail("Official split counts do not match expected counts.")

    print("\nOfficial split counts:")
    for split in SPLITS:
        total = 0
        print(f"\n  {split.upper()}:")
        for class_name in CLASS_NAMES:
            n = int(counts.get((split, class_name), 0))
            total += n
            print(f"    {class_name:<20}: {n}")
        print(f"    {'Total':<20}: {total}")


def validate_sources_exist(df: pd.DataFrame) -> None:
    missing = []

    for _, row in df.iterrows():
        metadata_name = Path(row["processed_image_path"]).name
        raw_path = RAW_DIR / metadata_name
        official_path = PROJECT_ROOT / row["official_source_path"]
        if not raw_path.exists() and not official_path.exists():
            missing.append(metadata_name)

    if missing:
        print("Missing source images:")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        fail("Cannot rebuild processed dataset because some source images are missing.")


def rebuild_processed_dataset(df: pd.DataFrame) -> None:
    for split in SPLITS:
        split_dir = OUTPUT_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    copied = 0
    used_official_fallback = 0

    for _, row in df.iterrows():
        metadata_name = Path(row["processed_image_path"]).name
        split = row["official_split"]
        class_name = row["official_class_name"]
        raw_path = RAW_DIR / metadata_name
        official_path = PROJECT_ROOT / row["official_source_path"]
        source = raw_path if raw_path.exists() else official_path
        if source == official_path:
            used_official_fallback += 1

        dest = OUTPUT_DIR / split / class_name / metadata_name
        shutil.copy2(source, dest)
        copied += 1

    print(f"\nCopied images: {copied}")
    print(f"Used official_split fallback sources: {used_official_fallback}")


def validate_processed_counts() -> None:
    rows = []
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            class_dir = OUTPUT_DIR / split / class_name
            count = sum(1 for p in class_dir.iterdir() if p.is_file())
            rows.append(
                {
                    "official_split": split,
                    "official_class_name": class_name,
                    "count": count,
                }
            )

    df = pd.DataFrame(rows)
    expanded = []
    for _, row in df.iterrows():
        for _ in range(int(row["count"])):
            expanded.append(
                {
                    "official_split": row["official_split"],
                    "official_class_name": row["official_class_name"],
                }
            )
    validate_expected_counts(pd.DataFrame(expanded), source="data/processed")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


if __name__ == "__main__":
    main()
