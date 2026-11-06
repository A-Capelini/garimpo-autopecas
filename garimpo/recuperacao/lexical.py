"""Baseline lexical: TF-IDF de n-gramas de caracteres (tolera abreviações)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from garimpo.dados import normalizar_texto


class BuscaLexical:
    def __init__(self, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self._matriz = None
        self._codigos: list[str] = []

    def indexar(self, codigos: list[str], textos: list[str]) -> None:
        self._codigos = list(codigos)
        self._matriz = self._vectorizer.fit_transform(textos)

    def buscar(self, consulta: str, k: int = 10) -> list[str]:
        if self._matriz is None:
            raise RuntimeError("chame indexar() antes de buscar()")
        vetor_consulta = self._vectorizer.transform([normalizar_texto(consulta)])
        scores = cosine_similarity(vetor_consulta, self._matriz)[0]
        ordem = np.argsort(-scores)[:k]
        return [self._codigos[i] for i in ordem]
