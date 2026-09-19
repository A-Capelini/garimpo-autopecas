"""Baseline lexical: TF-IDF de n-gramas de caractere (nao usa modelo de IA)."""
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    nome = "tfidf-char"

    def __init__(self, ngram_range=(3, 5), min_df=2):
        self._vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, min_df=min_df, sublinear_tf=True
        )
        self._X = None

    def fit(self, textos: Sequence[str]) -> None:
        self._X = self._vec.fit_transform(list(textos))

    def score_rows(self, linhas: Sequence[int]) -> np.ndarray:
        return (self._X[list(linhas)] @ self._X.T).toarray()

    def score_texts(self, consultas: Sequence[str]) -> np.ndarray:
        Q = self._vec.transform(list(consultas))
        return (Q @ self._X.T).toarray()
