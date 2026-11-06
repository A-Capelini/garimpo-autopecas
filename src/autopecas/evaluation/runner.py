"""Avaliacao de um recuperador em cada nivel de escala.

Regra (decisao 3):
- recuperador corpus-dependente (TF-IDF): reajusta o indice com SO os documentos do nivel;
- recuperador corpus-independente (embeddings): indexa o catalogo inteiro uma vez e
  fatia por nivel (o cosseno de um par nao muda com o resto do corpus).
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from autopecas.config import NIVEIS
from autopecas.data.levels import mascara_nivel
from autopecas.evaluation.gold import construir_gold
from autopecas.evaluation.metrics import avaliar

COLUNAS = [
    "retriever", "nivel", "n_docs", "n_consultas",
    "hit@1", "hit@5", "hit@10", "mrr", "seg_indexacao", "ms_por_consulta",
]


def _scores(retriever, consultas: np.ndarray, n_docs: int) -> np.ndarray:
    if len(consultas) == 0:
        return np.empty((0, n_docs))
    return retriever.score_rows(consultas)


def avaliar_niveis(
    df: pd.DataFrame,
    fabrica: Callable[[], object],
    niveis: Iterable[str] | None = None,
    coluna_texto: str = "texto_busca",
) -> pd.DataFrame:
    """`fabrica()` devolve um recuperador novo (sem ajuste). Retorna uma linha por nivel."""
    df = df.reset_index(drop=True)
    niveis = list(niveis) if niveis is not None else list(NIVEIS)
    modelo = fabrica()

    t_fit_unico = 0.0
    if not modelo.corpus_dependente:
        t0 = time.time()
        modelo.fit(df[coluna_texto].tolist())
        t_fit_unico = time.time() - t0

    linhas = []
    for nivel in niveis:
        mascara = mascara_nivel(df, nivel)
        if modelo.corpus_dependente:
            sub = df[mascara].reset_index(drop=True)
            r = fabrica()
            t0 = time.time()
            r.fit(sub[coluna_texto].tolist())
            t_fit = time.time() - t0
            gold = construir_gold(sub)
            t1 = time.time()
            m = avaliar(_scores(r, gold.consultas, len(sub)), gold)
            n_docs = len(sub)
        else:
            r, t_fit = modelo, t_fit_unico
            gold = construir_gold(df, mascara)
            t1 = time.time()
            m = avaliar(_scores(r, gold.consultas, len(df)), gold, mascara)
            n_docs = int(mascara.sum())
        m.update(
            retriever=r.nome,
            nivel=nivel,
            n_docs=n_docs,
            seg_indexacao=round(t_fit, 1),
            ms_por_consulta=round(1000 * (time.time() - t1) / max(len(gold), 1), 2),
        )
        linhas.append(m)
    return pd.DataFrame(linhas)[COLUNAS]
