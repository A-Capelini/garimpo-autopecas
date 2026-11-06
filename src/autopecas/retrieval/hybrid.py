"""Busca hibrida: funde um recuperador lexical e um semantico (embeddings).

Metodos de fusao
- "rrf"  : Reciprocal Rank Fusion. Cada fonte contribui peso / (k_rrf + posicao) para os
           documentos que estao no seu top-`profundidade`; os demais contribuem 0.
- "soma" : soma ponderada de scores normalizados (min-max por consulta):
           peso_lexical * lex + (1 - peso_lexical) * sem, com peso_lexical em [0, 1].

Regras de correcao
- A posicao (rank) e calculada DENTRO do corpus ajustado (no nivel, na avaliacao por nivel)
  e SEM a propria consulta (score_rows poe a linha consultada em -inf antes de ranquear).
- `fallback_lexical=True` (uso no app): se o Ollama falhar, devolve so o lexical e marca
  `degradado`. Na avaliacao fica False, para que uma falha nao vire resultado silencioso.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from autopecas.retrieval.ollama_embed import ErroOllama

DECIMAIS_EMPATE = 10  # mesmo criterio de evaluation.metrics


def _posicoes(S: np.ndarray) -> np.ndarray:
    """Posicao 1-based de cada documento no ranking decrescente da propria linha."""
    S = np.round(S, DECIMAIS_EMPATE)
    ordem = np.argsort(-S, axis=1, kind="stable")
    pos = np.empty_like(ordem)
    linhas = np.arange(S.shape[0])[:, None]
    pos[linhas, ordem] = np.arange(1, S.shape[1] + 1)
    return pos


def _minmax(S: np.ndarray) -> np.ndarray:
    fin = np.isfinite(S)
    lo = np.where(fin, S, np.inf).min(axis=1, keepdims=True)
    hi = np.where(fin, S, -np.inf).max(axis=1, keepdims=True)
    amp = np.where(hi - lo > 0, hi - lo, 1.0)
    return np.where(fin, (S - lo) / amp, 0.0)


class HybridRetriever:
    # O lexical precisa ser reajustado por nivel -> o hibrido tambem (os embeddings vem do
    # cache SQLite, entao reajustar nao chama o Ollama de novo).
    corpus_dependente = True

    def __init__(
        self,
        lexical,
        semantico,
        metodo: str = "rrf",
        peso_lexical: float = 3.0,
        peso_semantico: float = 1.0,
        k_rrf: int = 60,
        profundidade: int | None = 50,
        fallback_lexical: bool = False,
    ) -> None:
        if metodo not in ("rrf", "soma"):
            raise ValueError("metodo deve ser 'rrf' ou 'soma'")
        if metodo == "soma" and not 0.0 <= peso_lexical <= 1.0:
            raise ValueError("na soma, peso_lexical deve estar em [0, 1]")
        self.lexical, self.semantico = lexical, semantico
        self.metodo = metodo
        self.peso_lexical, self.peso_semantico = peso_lexical, peso_semantico
        self.k_rrf, self.profundidade = k_rrf, profundidade
        self.fallback_lexical = fallback_lexical
        self.degradado = False
        if metodo == "rrf":
            self.nome = f"hibrido-rrf[{peso_lexical:g}:{peso_semantico:g},k={k_rrf},d={profundidade}]"
        else:
            self.nome = f"hibrido-soma[w_lex={peso_lexical:g}]"

    # ---------- ajuste ----------
    def fit(self, textos: Sequence[str]) -> "HybridRetriever":
        textos = list(textos)
        self.degradado = False
        self.lexical.fit(textos)
        try:
            self.semantico.fit(textos)
        except ErroOllama:
            if not self.fallback_lexical:
                raise
            self.degradado = True
        return self

    # ---------- fusao ----------
    def _fundir(self, Sl: np.ndarray, Ss: np.ndarray) -> np.ndarray:
        if self.metodo == "soma":
            w = self.peso_lexical
            return w * _minmax(Sl) + (1.0 - w) * _minmax(Ss)
        fundido = np.zeros_like(Sl, dtype=np.float64)
        for S, w in ((Sl, self.peso_lexical), (Ss, self.peso_semantico)):
            pos = _posicoes(S)
            contrib = w / (self.k_rrf + pos)
            if self.profundidade:
                contrib = np.where(pos <= self.profundidade, contrib, 0.0)
            fundido += contrib
        return fundido

    # ---------- contrato Retriever ----------
    def score_rows(self, linhas: Sequence[int]) -> np.ndarray:
        linhas = np.asarray(list(linhas), dtype=int)
        idx = np.arange(len(linhas))
        Sl = np.asarray(self.lexical.score_rows(linhas), dtype=np.float64).copy()
        Sl[idx, linhas] = -np.inf  # a propria consulta nao entra no ranking
        if self.degradado:
            return Sl
        Ss = np.asarray(self.semantico.score_rows(linhas), dtype=np.float64).copy()
        Ss[idx, linhas] = -np.inf
        F = self._fundir(Sl, Ss)
        F[idx, linhas] = -np.inf
        return F

    def score_texts(self, consultas: Sequence[str]) -> np.ndarray:
        consultas = list(consultas)
        Sl = np.asarray(self.lexical.score_texts(consultas), dtype=np.float64)
        if self.degradado:
            return Sl
        try:
            Ss = np.asarray(self.semantico.score_texts(consultas), dtype=np.float64)
        except ErroOllama:
            if not self.fallback_lexical:
                raise
            self.degradado = True
            return Sl
        return self._fundir(Sl, Ss)
