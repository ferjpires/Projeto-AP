import pickle
import math

import numpy as np
from collections import Counter


class NumpyTfIdf:
    """
    Implementação de TF-IDF (Term Frequency - Inverse Document Frequency)
    Inspirada nos guiões das aulas práticas
    """

    def __init__(self, max_words=5000, analyzer="word", ngram_range=(1, 1)):
        self.max_words = max_words
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.vocab = {}
        self.idf = {}

    def _get_ngrams(self, text: str) -> list:
        if self.analyzer == "word":
            return text.split()
        text = f" {text} "
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams += [text[i : i + n] for i in range(len(text) - n + 1)]
        return ngrams

    def fit(self, texts):
        """
        Lê os textos de treino, constrói o vocabulário e calcula o IDF.
        """
        counter = Counter()
        doc_count = len(texts)
        word_doc_count = Counter()

        for text in texts:
            tokens = self._get_ngrams(text)
            counter.update(tokens)
            unique_tokens = set(tokens)
            word_doc_count.update(unique_tokens)

        most_common = counter.most_common(self.max_words)

        for i, (word, _count) in enumerate(most_common):
            self.vocab[word] = i
            self.idf[word] = math.log(doc_count / (1 + word_doc_count[word]))

    def transform(self, texts):
        """
        Converte uma lista de textos numa matriz TF-IDF (N_textos x Tamanho_Vocab)
        """
        tfidf_matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)

        for doc_idx, text in enumerate(texts):
            tokens = self._get_ngrams(text)
            token_count = len(tokens)

            if token_count == 0:
                continue

            # Contar frequências no documento
            token_counter = Counter(tokens)

            for token, count in token_counter.items():
                if token in self.vocab:
                    word_idx = self.vocab[token]
                    tf = count / token_count
                    tfidf_matrix[doc_idx, word_idx] = tf * self.idf[token]

        return tfidf_matrix

    def fit_transform(self, texts):
        """Treina o vocabulário e transforma logo os textos numa matriz."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        """Guarda o vocabulário e IDF num ficheiro pickle."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "max_words": self.max_words,
                    "analyzer": self.analyzer,
                    "ngram_range": self.ngram_range,
                },
                f,
            )

    def load(self, path: str):
        """Carrega um vocabulário previamente treinado."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.max_words = data["max_words"]
        self.analyzer = data.get("analyzer", "word")
        self.ngram_range = data.get("ngram_range", (1, 1))
