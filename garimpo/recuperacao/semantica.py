"""Busca semântica em memória (NumPy) e fusão híbrida por RRF."""

from __future__ import annotations

import numpy as np


class BuscaSemantica:
    def __init__(self) -> None:
        self._matriz: np.ndarray | None = None
        self._codigos: list[str] = []

    def indexar(self, codigos: list[str], matriz: np.ndarray) -> None:
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        self._matriz = matriz / np.where(normas == 0, 1, normas)
        self._codigos = list(codigos)
        self._posicao = {c: i for i, c in enumerate(self._codigos)}

    def reranquear(self, vetor: np.ndarray, candidatos: list[str], k: int = 10) -> list[str]:
        """Reordena candidatos (ex.: top-50 lexical) pela similaridade semântica."""
        if self._matriz is None:
            raise RuntimeError("chame indexar() antes de reranquear()")
        norma = np.linalg.norm(vetor) or 1.0
        idx = [self._posicao[c] for c in candidatos]
        scores = self._matriz[idx] @ (vetor / norma)
        ordem = np.argsort(-scores)[:k]
        return [candidatos[i] for i in ordem]

    def buscar_vetor(self, vetor: np.ndarray, k: int = 10) -> list[str]:
        if self._matriz is None:
            raise RuntimeError("chame indexar() antes de buscar_vetor()")
        norma = np.linalg.norm(vetor) or 1.0
        scores = self._matriz @ (vetor / norma)
        ordem = np.argsort(-scores)[:k]
        return [self._codigos[i] for i in ordem]


def fusao_rrf(
    rankings: list[list[str]], pesos: list[float] | None = None, k_rrf: int = 60
) -> list[str]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): soma peso/(k+posição)."""
    pesos = pesos or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, peso in zip(rankings, pesos):
        for posicao, codigo in enumerate(ranking, start=1):
            scores[codigo] = scores.get(codigo, 0.0) + peso / (k_rrf + posicao)
    return sorted(scores, key=scores.get, reverse=True)
