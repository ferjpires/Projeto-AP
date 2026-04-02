from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import nltk
from nltk.tokenize import word_tokenize

from src.tfidf import NumpyTfIdf
from src.stylometric_features import StylometricFeaturesExtractor


class Vectorizer(ABC):
    """Abstract base class for all vectorizers."""
    @abstractmethod
    def fit(self, texts, texts_raw=None):
        pass

    @abstractmethod
    def transform(self, texts, texts_raw=None):
        pass

    def fit_transform(self, texts, texts_raw=None):
        self.fit(texts, texts_raw)
        return self.transform(texts, texts_raw)

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

    def fit(self, texts, texts_raw=None):
        self.tfidf.fit(texts)
        return self

    def transform(self, texts, texts_raw=None):
        return self.tfidf.transform(texts)

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

    def fit(self, texts, texts_raw=None):
        self.tfidf.fit(texts)
        return self

    def transform(self, texts, texts_raw=None):
        return self.tfidf.transform(texts)

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

    def fit(self, texts, texts_raw):
        assert len(texts) == len(texts_raw), (
            f"texts ({len(texts)}) and texts_raw ({len(texts_raw)}) must have the same length"
        )

        self.char_vectorizer.fit(texts_raw)
        X_style = self.style_extractor.fit_transform(texts_raw)
        self.style_mean = X_style.mean(axis=0)
        self.style_std = X_style.std(axis=0) + 1e-8
        return self

    def transform(self, texts, texts_raw):
        assert len(texts) == len(texts_raw), (
            f"texts ({len(texts)}) and texts_raw ({len(texts_raw)}) must have the same length"
        )

        X_tfidf = self.char_vectorizer.transform(texts_raw)
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

FUNCTION_WORDS = [
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "no", "every", "each", "all",
    "both", "few", "many", "much", "several", "such", "what", "which", "whose",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "along", "until", "upon", "across", "against",
    "among", "around", "behind", "beside", "beyond", "despite", "down",
    "inside", "near", "off", "onto", "outside", "over", "past", "since",
    "toward", "towards", "throughout", "underneath", "unlike", "within", "without",
    "and", "but", "or", "nor", "for", "yet", "so", "because", "although",
    "though", "while", "whereas", "unless", "since", "if", "when", "where",
    "after", "before", "until", "once", "whether", "than", "that", "as",
    "i", "me", "we", "us", "you", "he", "him", "she", "it", "they", "them",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "who", "whom", "whose", "which", "that", "what", "whoever", "whatever",
    "one", "ones", "something", "anything", "nothing", "everything",
    "someone", "anyone", "everyone", "nobody",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "not", "also", "very", "often", "however", "too", "usually", "really",
    "already", "always", "never", "sometimes", "still", "just", "only",
    "quite", "rather", "almost", "enough", "even", "perhaps", "certainly",
    "probably", "actually", "apparently", "basically", "clearly", "definitely",
    "essentially", "furthermore", "generally", "hence", "indeed", "moreover",
    "namely", "nevertheless", "nonetheless", "notably", "otherwise",
    "particularly", "precisely", "primarily", "respectively", "significantly",
    "similarly", "specifically", "subsequently", "therefore", "thus",
    "ultimately", "virtually", "wherein", "whereby", "meanwhile",
    "here", "there", "then", "now", "how", "why", "where", "when",
    "more", "most", "less", "least", "well", "back", "else", "far",
]

FUNCTION_WORDS = list(dict.fromkeys(FUNCTION_WORDS))


class ForensicVectorizer(Vectorizer):
    """Topic-independent vectorizer using POS n-grams, function word frequencies, and stylometric features."""
    def __init__(self, max_words=1000, **kwargs):
        self.pos_tfidf = NumpyTfIdf(max_words=max_words, analyzer="word")
        self.function_words = FUNCTION_WORDS
        self.style_extractor = StylometricFeaturesExtractor()
        self.func_mean = None
        self.func_std = None
        self.style_mean = None
        self.style_std = None

    def _extract_pos_sequences(self, texts_raw):
        """Convert texts to POS tag bigram/trigram sequences."""
        pos_sequences = []
        for text in texts_raw:
            tokens = word_tokenize(str(text))
            tags = nltk.pos_tag(tokens)
            tag_seq = [t[1] for t in tags]
            bigrams = [f"{tag_seq[i]}_{tag_seq[i+1]}" for i in range(len(tag_seq) - 1)]
            trigrams = [f"{tag_seq[i]}_{tag_seq[i+1]}_{tag_seq[i+2]}" for i in range(len(tag_seq) - 2)]
            pos_sequences.append(" ".join(bigrams + trigrams))
        return pos_sequences

    def _extract_function_word_freqs(self, texts_raw):
        """Relative frequency of each function word."""
        matrix = np.zeros((len(texts_raw), len(self.function_words)))
        for i, text in enumerate(texts_raw):
            words = word_tokenize(str(text).lower())
            total = len(words) if words else 1
            word_counts = Counter(words)
            for j, fw in enumerate(self.function_words):
                matrix[i, j] = word_counts.get(fw, 0) / total
        return matrix

    def fit(self, texts, texts_raw):
        pos_seqs = self._extract_pos_sequences(texts_raw)
        self.pos_tfidf.fit(pos_seqs)

        func_feats = self._extract_function_word_freqs(texts_raw)
        self.func_mean = func_feats.mean(axis=0)
        self.func_std = func_feats.std(axis=0) + 1e-8

        style_feats = self.style_extractor.fit_transform(texts_raw)
        self.style_mean = style_feats.mean(axis=0)
        self.style_std = style_feats.std(axis=0) + 1e-8
        return self

    def transform(self, texts, texts_raw):
        X_pos = self.pos_tfidf.transform(self._extract_pos_sequences(texts_raw))
        X_func = self._extract_function_word_freqs(texts_raw)
        X_func = (X_func - self.func_mean) / self.func_std
        X_style = self.style_extractor.transform(texts_raw)
        X_style = (X_style - self.style_mean) / self.style_std
        return np.hstack([X_pos, X_func, X_style])

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "func_mean": self.func_mean,
                "func_std": self.func_std,
                "style_mean": self.style_mean,
                "style_std": self.style_std,
                "pos_tfidf_path": path + ".pos.pkl",
            }, f)
        self.pos_tfidf.save(path + ".pos.pkl")

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.func_mean = data["func_mean"]
        self.func_std = data["func_std"]
        self.style_mean = data["style_mean"]
        self.style_std = data["style_std"]
        self.pos_tfidf.load(data["pos_tfidf_path"])


def create_vectorizer(vectorizer_type: str, **kwargs) -> Vectorizer:
    if vectorizer_type == "word":
        return WordVectorizer(**kwargs)
    elif vectorizer_type == "char":
        return CharNgramVectorizer(**kwargs)
    elif vectorizer_type == "stylometric":
        return StylometricVectorizer(**kwargs)
    elif vectorizer_type == "forensic":
        return ForensicVectorizer(**kwargs)
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
