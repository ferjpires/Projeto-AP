import re
import pandas as pd

# Tirado dos tpc's, sem dependencias externas
STOP_WORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "in", "on", "with", "for", "of", "it", "this", "that"}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in STOP_WORDS]
    
    return " ".join(tokens)

def clean_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    df_clean = df.copy()
    df_clean[text_column] = df_clean[text_column].apply(clean_text)
    return df_clean