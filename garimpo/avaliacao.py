"""Gabarito por GTIN e métricas de recuperação: Hit@k, Recall@k, MRR.

Gabarito (relato, seção 4.3): cada linha de um fornecedor com GTIN válido
que aparece em >1 fornecedor é uma consulta; as linhas de outros
fornecedores com o mesmo GTIN são os resultados relevantes.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from garimpo.dados import gtin_valido


def construir_gabarito(df: pd.DataFrame) -> dict[str, set[str]]:
    """codigo (consulta) -> conjunto de codigos relevantes (outro fornecedor, mesmo GTIN)."""
    validos = df[df["gtin"].apply(gtin_valido)]
    por_gtin: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, linha in validos.iterrows():
        por_gtin[linha["gtin"]].append((linha["codigo"], linha["fornecedor"]))

    gabarito: dict[str, set[str]] = {}
    for pares in por_gtin.values():
        if len(pares) < 2:
            continue
        for codigo, fornecedor in pares:
            relevantes = {c for c, f in pares if f != fornecedor}
            if relevantes:
                gabarito[codigo] = relevantes
    return gabarito


def hit_at_k(resultados: list[str], relevantes: set[str]) -> bool:
    return any(r in relevantes for r in resultados)


def recall_at_k(resultados: list[str], relevantes: set[str]) -> float:
    if not relevantes:
        return 0.0
    return len(set(resultados) & relevantes) / len(relevantes)


def reciprocal_rank(resultados: list[str], relevantes: set[str]) -> float:
    for posicao, codigo in enumerate(resultados, start=1):
        if codigo in relevantes:
            return 1.0 / posicao
    return 0.0


def avaliar(gabarito: dict[str, set[str]], buscar) -> dict[str, float]:
    """`buscar(consulta_codigo) -> list[codigo]` já ordenado por relevância (top-k)."""
    hits1 = hits5 = hits10 = 0
    soma_rr = 0.0
    n = len(gabarito)
    if n == 0:
        return {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0, "n_consultas": 0}

    for consulta, relevantes in gabarito.items():
        resultados = buscar(consulta)
        hits1 += hit_at_k(resultados[:1], relevantes)
        hits5 += hit_at_k(resultados[:5], relevantes)
        hits10 += hit_at_k(resultados[:10], relevantes)
        soma_rr += reciprocal_rank(resultados[:10], relevantes)

    return {
        "hit@1": hits1 / n,
        "hit@5": hits5 / n,
        "hit@10": hits10 / n,
        "mrr": soma_rr / n,
        "n_consultas": n,
    }
