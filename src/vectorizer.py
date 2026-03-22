"""
Vectorizer hierarchy for text feature extraction.
Supports word TF-IDF, char n-gram TF-IDF, and stylometric features.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.tfidf import NumpyTfIdf
from src.stylometric_features import StylometricFeaturesExtractor


class Vectorizer(ABC):
    """Abstract base class for all vectorizers."""

    @abstractmethod
    def fit(self, texts_clean, texts_raw):
        pass

    @abstractmethod
    def transform(self, texts_clean, texts_raw):
        pass

    def fit_transform(self, texts_clean, texts_raw):
        self.fit(texts_clean, texts_raw)
        return self.transform(texts_clean, texts_raw)

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class WordVectorizer(Vectorizer):
    """Word-level TF-IDF vectorizer"""

    def __init__(self, max_words=5000):
        self.tfidf = NumpyTfIdf(max_words=max_words, analyzer="word")

    def fit(self, texts_clean, texts_raw=None):
        self.tfidf.fit(texts_clean)
        return self

    def transform(self, texts_clean, texts_raw=None):
        return self.tfidf.transform(texts_clean)

    def save(self, path):
        self.tfidf.save(path)

    def load(self, path):
        self.tfidf.load(path)


class CharNgramVectorizer(Vectorizer):
    """Character n-gram TF-IDF vectorizer"""

    def __init__(self, max_words=5000, ngram_range=(2, 3)):
        self.tfidf = NumpyTfIdf(
            max_words=max_words, analyzer="char", ngram_range=ngram_range
        )

    def fit(self, texts_clean, texts_raw=None):
        self.tfidf.fit(texts_clean)
        return self

    def transform(self, texts_clean, texts_raw=None):
        return self.tfidf.transform(texts_clean)

    def save(self, path):
        self.tfidf.save(path)

    def load(self, path):
        self.tfidf.load(path)


class StylometricVectorizer(Vectorizer):
    def __init__(self, max_words=5000, ngram_range=(2, 3)):
        self.char_vectorizer = CharNgramVectorizer(
            max_words=max_words, ngram_range=ngram_range
        )
        self.style_extractor = StylometricFeaturesExtractor()
        self.style_mean = None
        self.style_std = None

    def fit(self, texts_clean, texts_raw):
        assert len(texts_clean) == len(texts_raw), (
            f"texts_clean ({len(texts_clean)}) and texts_raw ({len(texts_raw)}) must have the same length"
        )

        self.char_vectorizer.fit(texts_raw)
        X_style = self.style_extractor.fit_transform(texts_raw)
        self.style_mean = X_style.mean(axis=0)
        self.style_std = X_style.std(axis=0) + 1e-8
        return self

    def transform(self, texts_clean, texts_raw):
        assert len(texts_clean) == len(texts_raw), (
            f"texts_clean ({len(texts_clean)}) and texts_raw ({len(texts_raw)}) must have the same length"
        )

        X_tfidf = self.char_vectorizer.transform(texts_clean)
        X_style = self.style_extractor.transform(texts_raw)
        X_style = (X_style - self.style_mean) / self.style_std
        return np.hstack([X_tfidf, X_style])

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "char_vectorizer_path": path + ".char.pkl",
                    "style_mean": self.style_mean,
                    "style_std": self.style_std,
                },
                f,
            )
        self.char_vectorizer.save(path + ".char.pkl")

    def load(self, path):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.char_vectorizer.load(data["char_vectorizer_path"])
        self.style_mean = data["style_mean"]
        self.style_std = data["style_std"]


def create_vectorizer(vectorizer_type: str, **kwargs) -> Vectorizer:
    if vectorizer_type == "word":
        return WordVectorizer(**kwargs)
    elif vectorizer_type == "char":
        return CharNgramVectorizer(**kwargs)
    elif vectorizer_type == "stylometric":
        return StylometricVectorizer(**kwargs)
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
