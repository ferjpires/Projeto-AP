import re
import numpy as np
from typing import List

FEATURE_NAMES = [
    "avg_word_length",
    "type_token_ratio",
    "avg_sentence_length",
    "sentence_length_std",
    "comma_rate",
    "avg_paragraph_length",
    "parenthesis_rate",
    "first_person_rate",
    "passive_voice_approx",
    "hedge_rate",
    "discourse_marker_rate",
    "colon_rate",
    "dash_rate",
    "quotes_rate",
    "word_density",
    "transition_density",
    "lexical_density",
    "sent_len_cv",
    "log_word_count"
]

N_FEATURES = len(FEATURE_NAMES)


class StylometricFeaturesExtractor:
    def __init__(self):
        self.HEDGERS = [
            "however", "although", "while", "whereas", "nevertheless",
            "nonetheless", "it is worth noting", "it should be noted"
        ]

        self.DISCOURSE = [
            "furthermore", "moreover", "in conclusion", "in summary",
            "notably", "specifically", "in particular", "as a result"
        ]

        self.TRANSITIONS = [
            "furthermore", "moreover", "however", "nevertheless", "nonetheless",
            "whereas", "while", "although", "in addition", "consequently",
            "therefore", "thus"
        ]

    def fit(self, texts: List[str]):
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        features = []
        for text in texts:
            feats = self._extract_features(str(text))
            features.append(feats)
        return np.array(features, dtype=np.float32)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.transform(texts)

    def _extract_features(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * N_FEATURES

        words = re.findall(r'\b\w+\b', text)
        lower_text = text.lower()
        num_words = max(len(words), 1)
        num_chars = max(len(text), 1)

        avg_word_length = sum(len(w) for w in words) / num_words

        unique_words = len(set(w.lower() for w in words))
        type_token_ratio = unique_words / num_words

        # 1. Sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        sentence_lengths = [max(len(s.split()), 1) for s in sentences]

        avg_sentence_length = sum(sentence_lengths) / num_sentences
        sentence_length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

        # 2. Punctuation
        comma_rate = text.count(',') / num_words

        # 3. Structural
        paragraphs = text.split('\n')
        paragraphs = [p for p in paragraphs if p.strip()]
        num_paragraphs = max(len(paragraphs), 1)
        avg_paragraph_length = num_words / num_paragraphs

        parenthesis_count = text.count('(') + text.count(')')
        parenthesis_rate = parenthesis_count / num_words

        # 4. Stylistic markers
        first_person_words = {"i", "we", "my", "mine", "our", "ours", "us", "me"}
        first_person_count = sum(1 for w in (w.lower() for w in words) if w in first_person_words)
        first_person_rate = first_person_count / num_words

        passive_matches = len(re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', lower_text))
        passive_voice_approx = passive_matches / num_sentences

        # 5. Hedgers and discourse markers
        hedge_count = sum(lower_text.count(h) for h in self.HEDGERS)
        hedge_rate = hedge_count / num_sentences

        discourse_count = sum(lower_text.count(d) for d in self.DISCOURSE)
        discourse_marker_rate = discourse_count / num_sentences

        # 6. Other symbols
        colon_rate = text.count(':') / num_words
        dash_rate = text.count('-') / num_words
        quotes_rate = (text.count('"') + text.count("'")) / num_words

        word_density = len(words) / num_chars

        # 7. New features
        transition_count = sum(lower_text.count(t) for t in self.TRANSITIONS)
        transition_density = transition_count / num_sentences

        lexical_density = sum(1 for w in words if len(w) > 6) / num_words

        sent_len_cv = (sentence_length_std / avg_sentence_length) if avg_sentence_length > 0 else 0.0

        log_word_count = np.log1p(num_words)

        return [
            avg_word_length,
            type_token_ratio,
            avg_sentence_length,
            sentence_length_std,
            comma_rate,
            avg_paragraph_length,
            parenthesis_rate,
            first_person_rate,
            passive_voice_approx,
            hedge_rate,
            discourse_marker_rate,
            colon_rate,
            dash_rate,
            quotes_rate,
            word_density,
            transition_density,
            lexical_density,
            sent_len_cv,
            log_word_count
        ]
