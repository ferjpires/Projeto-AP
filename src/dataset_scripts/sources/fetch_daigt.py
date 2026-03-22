import pandas as pd

from utils import truncate_text

LABEL_MAP = {
    "persuade_corpus": "Human",
    "train_essays": "Human",
    "chat_gpt_moth": "OpenAI",
    "radekgpt4": "OpenAI",
    "radek_500": "OpenAI",
    "kingki19_palm": "Google",
    "palm-text-bison1": "Google",
    "llama2_chat": "Meta",
    "llama_70b_v1": "Meta",
    "NousResearch/Llama-2-7b-chat-hf": "Meta",
    "darragh_claude_v6": "Anthropic",
    "darragh_claude_v7": "Anthropic",
}


def fetch_daigt(raw_path="../../data/raw/dataset_kaggle_daigt.csv") -> pd.DataFrame:
    print(f"Loading DAIGT dataset from {raw_path}...")

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"WARNING: Could not find {raw_path}")
        return pd.DataFrame(columns=["Text", "Label"])

    df["Label"] = df["source"].map(LABEL_MAP)

    classes_oficiais = ["Human", "OpenAI", "Meta", "Google", "Anthropic"]
    df = df[df["Label"].isin(classes_oficiais)].copy()

    df["Text"] = df["text"].apply(lambda x: truncate_text(str(x), max_words=120))

    df = df[["Text", "Label"]].dropna().reset_index(drop=True)

    print(f"  Loaded {len(df)} samples from DAIGT")
    print(f"  Distribution: {df['Label'].value_counts().to_dict()}")

    return df
