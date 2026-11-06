"""Contrato comum dos recuperadores."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Retriever(Protocol):
    nome: str
    # True: os scores dependem do corpus ajustado (ex.: IDF do TF-IDF) -> reajustar por nivel.
    # False: o score de um par nao depende dos demais documentos (ex.: cosseno de embeddings)
    #        -> indexar o catalogo inteiro uma vez e fatiar por nivel.
    corpus_dependente: bool

    def fit(self, textos: Sequence[str]) -> "Retriever | None": ...
    def score_rows(self, linhas: Sequence[int]) -> np.ndarray: ...
    def score_texts(self, consultas: Sequence[str]) -> np.ndarray: ...
