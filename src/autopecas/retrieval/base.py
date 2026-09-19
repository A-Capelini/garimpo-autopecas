"""Contrato comum dos recuperadores (lexical, embeddings, hibrido...)."""
from typing import Protocol, Sequence

import numpy as np


class Retriever(Protocol):
    nome: str

    def fit(self, textos: Sequence[str]) -> None:
        """Indexa o corpus (uma string por linha do catalogo)."""

    def score_rows(self, linhas: Sequence[int]) -> np.ndarray:
        """Scores (n_linhas x n_docs) usando linhas do proprio corpus como consulta."""

    def score_texts(self, consultas: Sequence[str]) -> np.ndarray:
        """Scores (n_consultas x n_docs) para textos livres (usado pela interface)."""
