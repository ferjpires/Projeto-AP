import pickle

import numpy as np
from collections import Counter


class NumpyBagOfWords:
    """
    Implementação de Bag of Words
    Inspirada nos guiões das aulas práticas
    """

    def __init__(self, max_words=5000):
        self.max_words = max_words
        self.vocab = {}
        
    def fit(self, texts):
        """
        Lê os textos de treino e constrói o vocabulário com as palavras mais frequentes.
        """
        counter = Counter()
        
        for text in texts:
            tokens = text.split()
            counter.update(tokens)
            
        most_common = counter.most_common(self.max_words)
        
        for i, (word, _count) in enumerate(most_common):
            self.vocab[word] = i
            
    def transform(self, texts):
        """
        Converte uma lista de textos numa matriz matemática Numpy (N_textos x Tamanho_Vocab)
        """
        bow_matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        
        for doc_idx, text in enumerate(texts):
            tokens = text.split()
            
            for token in tokens:
                if token in self.vocab:
                    word_idx = self.vocab[token]
                    bow_matrix[doc_idx, word_idx] += 1
                    
        return bow_matrix
        
    def fit_transform(self, texts):
        """Treina o vocabulário e transforma logo os textos numa matriz."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        """Guarda o vocabulário num ficheiro pickle."""
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "max_words": self.max_words}, f)

    def load(self, path: str):
        """Carrega um vocabulário previamente treinado."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.max_words = data["max_words"]