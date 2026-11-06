"""Ablacao do baseline lexical: o que explica a diferenca entre o pipeline oficial e o do Vitor?

Fatores (2 niveis cada -> 8 celulas):
  normalizacao : oficial (mantem '/' e '>') | simples (todo simbolo vira espaco)
  min_df       : 2 (oficial) | 1 (Vitor)
  sublinear_tf : True (oficial) | False (Vitor)
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

import pandas as pd

from autopecas.data.quality import normalizar_texto
from autopecas.evaluation.runner import avaliar_niveis
from autopecas.retrieval.lexical import TfidfRetriever

NORMALIZACOES = {"oficial": True, "simples": False}


def ablacao_lexical(df: pd.DataFrame, niveis: Iterable[str] | None = None) -> pd.DataFrame:
    """`df` precisa de `descricao`, `fornecedor` e `gtin_chave`. Uma linha por celula x nivel."""
    partes = []
    for (norm, manter), min_df, sub in product(NORMALIZACOES.items(), (2, 1), (True, False)):
        d = df.copy()
        d["texto_busca"] = d["descricao"].map(lambda t: normalizar_texto(t, manter_simbolos=manter))
        tab = avaliar_niveis(d, lambda: TfidfRetriever(min_df=min_df, sublinear_tf=sub), niveis)
        tab.insert(0, "sublinear_tf", sub)
        tab.insert(0, "min_df", min_df)
        tab.insert(0, "normalizacao", norm)
        partes.append(tab.drop(columns=["retriever", "seg_indexacao", "ms_por_consulta"]))
    return pd.concat(partes, ignore_index=True)
