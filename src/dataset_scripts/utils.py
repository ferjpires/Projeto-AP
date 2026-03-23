import pandas as pd
import csv

def truncate_text(text: str, max_words: int = 120) -> str:
    return " ".join(str(text).split()[:max_words])


def balance_classes(df, label_col="Label", random_state=42, max_per_class=500) -> pd.DataFrame:
    sampled = [
        g.sample(n=min(len(g), max_per_class), random_state=random_state)
        for _, g in df.groupby(label_col)
    ]
    return pd.concat(sampled, ignore_index=True)


def deduplicate(df, text_col="Text") -> pd.DataFrame:
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)


def save_dataset(df, path, prefix="DATASET") -> None:
    df = df.copy()
    df["ID"] = [f"{prefix}-{i + 1}" for i in range(len(df))]
    df_final = df[["ID", "Text", "Label"]]
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df_final.to_csv(path, sep=";", index=False, quoting=csv.QUOTE_ALL)
    return df_final