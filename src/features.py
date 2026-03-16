import pickle
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_words: int = 20000):
        self.max_words = max_words
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def fit(self, texts: list[str]):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        most_common = counter.most_common(self.max_words)

        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str) -> list[int]:
        return [self.word2idx.get(word, 1) for word in text.split()]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]


def texts_to_sequences(texts: list[str], vocab: Vocabulary, max_len: int = 200) -> np.ndarray:
    """
    Converte uma lista de textos limpos em sequências de IDs com padding.
    """
    sequences = np.zeros((len(texts), max_len), dtype=np.int64)

    for i, text in enumerate(texts):
        ids = vocab.encode(text)
        length = min(len(ids), max_len)
        sequences[i, :length] = ids[:length]

    return sequences


def load_glove_embeddings(
    glove_path: str, vocab: Vocabulary, embedding_dim: int = 100
) -> torch.Tensor:
    """
    Carrega embeddings GloVe de um ficheiro .txt e cria uma matriz
    alinhada com o vocabulário do projecto.
    """
    embedding_matrix = np.random.normal(0, 0.1, (len(vocab), embedding_dim)).astype(np.float32)
    embedding_matrix[0] = np.zeros(embedding_dim)

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab.word2idx:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == embedding_dim:
                    embedding_matrix[vocab.word2idx[word]] = vec
                    found += 1

    print(f"GloVe: {found}/{len(vocab)} palavras encontradas ({100*found/len(vocab):.1f}%)")
    return torch.tensor(embedding_matrix)


class TextDataset(Dataset):
    """
    PyTorch Dataset para classificação de texto.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
