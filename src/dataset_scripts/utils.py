"""Utility functions for dataset processing."""

import pandas as pd


def truncate_text(text: str, max_words: int = 120) -> str:
    """Truncate text to a maximum number of words."""
    return " ".join(str(text).split()[:max_words])


def balance_classes(df, label_col="Label", random_state=42) -> pd.DataFrame:
    """Balance classes by undersampling to the minority class size."""
    min_count = df[label_col].value_counts().min()
    return (
        df.groupby(label_col)
        .sample(n=min_count, random_state=random_state)
        .reset_index(drop=True)
    )


def deduplicate(df, text_col="Text") -> pd.DataFrame:
    """Remove duplicate texts."""
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)


def save_dataset(df, path, prefix="DATASET") -> None:
    """Save dataset with ID column and standard columns."""
    df = df.copy()
    df["ID"] = [f"{prefix}-{i + 1}" for i in range(len(df))]
    df_final = df[["ID", "Text", "Label"]]

    import os

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df_final.to_csv(path, sep=";", index=False)

    return df_final
