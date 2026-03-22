import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in STOP_WORDS
    ]

    return " ".join(tokens)


def clean_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    df_clean = df.copy()
    df_clean[text_column] = df_clean[text_column].apply(clean_text)
    return df_clean
