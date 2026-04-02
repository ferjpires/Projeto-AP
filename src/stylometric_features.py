import re
from collections import Counter

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
    "hapax_ratio",
    "dis_legomena_ratio",
    "yules_k",
    "sentence_starter_entropy",
    "unique_bigram_ratio",
    "flesch_reading_ease",
    "word_length_entropy",
    "punct_entropy",
]

N_FEATURES = len(FEATURE_NAMES)


class StylometricFeaturesExtractor:
    def __init__(self):
        self.HEDGERS = [
            "however", "although", "while", "whereas", "nevertheless",
            "nonetheless", "it is worth noting", "it should be noted",
            "arguably", "potentially", "seemingly", "presumably",
            "it is important to note", "one might argue", "to some extent",
            "it could be argued", "perhaps", "possibly", "likely",
            "generally speaking", "in many cases", "broadly speaking",
            "it is widely recognized", "to a certain degree",
            "in some respects",
        ]

        self.DISCOURSE = [
            "furthermore", "moreover", "in conclusion", "in summary",
            "notably", "specifically", "in particular", "as a result",
            "for instance", "for example", "in other words",
            "on the other hand", "in contrast", "by comparison",
            "as mentioned", "it is clear that",
            "overall", "ultimately", "to summarize", "in essence",
            "additionally", "equally important", "more importantly",
            "first and foremost", "last but not least",
        ]

        self.TRANSITIONS = [
            "furthermore", "moreover", "however", "nevertheless", "nonetheless",
            "whereas", "while", "although", "in addition", "consequently",
            "therefore", "thus", "accordingly", "hence", "meanwhile",
            "subsequently", "in contrast", "on the contrary",
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

        transition_count = sum(lower_text.count(t) for t in self.TRANSITIONS)
        transition_density = transition_count / num_sentences

        lexical_density = sum(1 for w in words if len(w) > 6) / num_words

        sent_len_cv = (sentence_length_std / avg_sentence_length) if avg_sentence_length > 0 else 0.0

        # Hapax & Dis Legomena (vocabulary richness)
        word_freq = Counter(w.lower() for w in words)
        hapax_ratio = sum(1 for v in word_freq.values() if v == 1) / num_words
        dis_legomena_ratio = sum(1 for v in word_freq.values() if v == 2) / num_words

        # Yule's K (vocabulary richness, size-independent)
        freq_spectrum = Counter(word_freq.values())
        N = num_words
        M = sum(i * i * freq_spectrum[i] for i in freq_spectrum)
        yules_k = 10000 * (M - N) / (N * N) if N > 1 else 0.0

        # Sentence starter entropy (LLMs are more repetitive)
        starters = [s.strip().split()[0].lower() for s in sentences if s.strip() and s.strip().split()]
        if len(starters) > 1:
            starter_freq = Counter(starters)
            starter_probs = np.array(list(starter_freq.values()), dtype=np.float64) / len(starters)
            sentence_starter_entropy = float(-np.sum(starter_probs * np.log2(starter_probs + 1e-12)))
        else:
            sentence_starter_entropy = 0.0

        # Unique bigram ratio (repetition measure)
        lower_words = [w.lower() for w in words]
        if len(lower_words) > 1:
            bigrams = [(lower_words[i], lower_words[i + 1]) for i in range(len(lower_words) - 1)]
            unique_bigram_ratio = len(set(bigrams)) / len(bigrams)
        else:
            unique_bigram_ratio = 1.0

        # Flesch Reading Ease
        syllable_count = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in words)
        flesch_reading_ease = (
            206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllable_count / num_words)
        )

        # Word length entropy
        word_lengths = [len(w) for w in words]
        wl_counts = np.histogram(word_lengths, bins=range(1, 21))[0]
        wl_probs = wl_counts / max(wl_counts.sum(), 1)
        word_length_entropy = float(-np.sum(wl_probs * np.log2(wl_probs + 1e-12)))

        # Punctuation distribution entropy
        punct_chars = '.,;:!?-()[]"\''
        punct_counts = np.array([text.count(p) for p in punct_chars], dtype=np.float64)
        punct_total = max(punct_counts.sum(), 1)
        punct_probs = punct_counts / punct_total
        punct_entropy = float(-np.sum(punct_probs * np.log2(punct_probs + 1e-12)))

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
            hapax_ratio,
            dis_legomena_ratio,
            yules_k,
            sentence_starter_entropy,
            unique_bigram_ratio,
            flesch_reading_ease,
            word_length_entropy,
            punct_entropy,
        ]
