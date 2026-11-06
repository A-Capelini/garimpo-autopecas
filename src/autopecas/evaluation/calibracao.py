"""Calibracao do hibrido com divisao por GTIN.

1. Varre normalizacao x min_df x fusao (RRF e soma normalizada); sublinear_tf fixo em True
   (a ablacao mostrou efeito desprezivel).
2. Escolhe, SO nas consultas de calibracao do nivel de selecao, o melhor lexical puro e o
   melhor hibrido (MRR; desempate por Hit@1).
3. Compara os dois nas consultas de TESTE, com IC95% pareado (bootstrap) da diferenca.
Regra de adocao: o hibrido e adotado se nao perder do lexical, no teste do nivel de selecao,
nem em Hit@1 nem em MRR (estimativa pontual).
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from autopecas.data.levels import mascara_nivel
from autopecas.data.quality import normalizar_texto
from autopecas.evaluation.gold import construir_gold
from autopecas.evaluation.metrics import metricas_de_posicoes, primeira_posicao
from autopecas.evaluation.split import mascara_calibracao
from autopecas.retrieval.hybrid import fundir_rrf, fundir_soma, posicoes
from autopecas.retrieval.lexical import TfidfRetriever

NORMALIZACOES = {"oficial": True, "simples": False}  # nome -> manter_simbolos
PESOS_RRF = ((1, 1), (2, 1), (3, 1), (1, 2))  # (peso_lexical, peso_semantico)
KS_RRF = (1, 5, 10, 20, 60)
PESOS_SOMA = (0.5, 0.6, 0.7, 0.8, 0.9)
PROFUNDIDADE = 50
COLS_METRICA = ["n_consultas", "hit@1", "hit@5", "hit@10", "mrr", "mrr@10"]


def rotulo_rrf(wl, ws, k) -> str:
    return f"rrf[{wl}:{ws},k={k}]"


def rotulo_soma(w) -> str:
    return f"soma[w_lex={w}]"


def _com_self_menos_inf(S: np.ndarray, consultas: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64).copy()
    S[np.arange(len(consultas)), consultas] = -np.inf
    return S


def calcular_posicoes(
    df: pd.DataFrame,
    fabrica_semantico: Callable[[], object],
    niveis: Iterable[str],
    normalizacoes=("oficial", "simples"),
    min_dfs=(2, 1),
    pesos_rrf=PESOS_RRF,
    ks_rrf=KS_RRF,
    pesos_soma=PESOS_SOMA,
    profundidade=PROFUNDIDADE,
) -> tuple[dict, dict]:
    """Devolve (posicoes, mascaras_cal).

    posicoes[(norm, min_df, nivel, config)] = array com a posicao do 1o equivalente por consulta
    config: 'lexical' | 'semantico' | 'rrf[..]' | 'soma[..]' (min_df=0 em 'semantico').
    mascaras_cal[nivel] = bool por consulta (True = calibracao).
    """
    df = df.reset_index(drop=True)
    posicoes_out: dict = {}
    cal_por_nivel: dict = {}
    for norm in normalizacoes:
        d = df.copy()
        manter = NORMALIZACOES[norm]
        d["texto_busca"] = d["descricao"].map(lambda t: normalizar_texto(t, manter_simbolos=manter))
        for nivel in niveis:
            sub = d[mascara_nivel(d, nivel)].reset_index(drop=True)
            gold = construir_gold(sub)
            if len(gold) == 0:
                continue
            cal_por_nivel[nivel] = mascara_calibracao(sub["gtin_chave"].to_numpy()[gold.consultas])
            sem = fabrica_semantico()
            sem.fit(sub["texto_busca"].tolist())
            Ss = _com_self_menos_inf(sem.score_rows(gold.consultas), gold.consultas)
            Ps = posicoes(Ss)
            posicoes_out[(norm, 0, nivel, "semantico")] = primeira_posicao(Ss, gold)
            for min_df in min_dfs:
                lex = TfidfRetriever(min_df=min_df)
                lex.fit(sub["texto_busca"].tolist())
                Sl = _com_self_menos_inf(lex.score_rows(gold.consultas), gold.consultas)
                Pl = posicoes(Sl)
                posicoes_out[(norm, min_df, nivel, "lexical")] = primeira_posicao(Sl, gold)
                for wl, ws in pesos_rrf:
                    for k in ks_rrf:
                        F = _com_self_menos_inf(fundir_rrf(Pl, Ps, wl, ws, k, profundidade), gold.consultas)
                        posicoes_out[(norm, min_df, nivel, rotulo_rrf(wl, ws, k))] = primeira_posicao(F, gold)
                for w in pesos_soma:
                    F = _com_self_menos_inf(fundir_soma(Sl, Ss, w), gold.consultas)
                    posicoes_out[(norm, min_df, nivel, rotulo_soma(w))] = primeira_posicao(F, gold)
    return posicoes_out, cal_por_nivel


def tabela(posicoes_out: dict, cal_por_nivel: dict) -> pd.DataFrame:
    linhas = []
    for (norm, min_df, nivel, config), pos in posicoes_out.items():
        cal = cal_por_nivel[nivel]
        for nome, m in (("calibracao", cal), ("teste", ~cal)):
            linhas.append({"normalizacao": norm, "min_df": min_df, "nivel": nivel, "config": config,
                           "split": nome, **metricas_de_posicoes(pos[m])})
    return pd.DataFrame(linhas)


def _melhor(tab: pd.DataFrame, nivel: str, so_hibrido: bool, so_lexical: bool = False) -> pd.Series:
    t = tab[(tab["nivel"] == nivel) & (tab["split"] == "calibracao")]
    if so_lexical:
        t = t[t["config"] == "lexical"]
    elif so_hibrido:
        t = t[~t["config"].isin(["lexical", "semantico"])]
    return t.sort_values(["mrr", "hit@1"], ascending=False, kind="stable").iloc[0]


def _bootstrap(delta: np.ndarray, n_boot: int, semente: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    medias = delta[idx].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def comparar_no_teste(posicoes_out, cal_por_nivel, tab, niveis, nivel_selecao, n_boot=2000) -> dict:
    lex = _melhor(tab, nivel_selecao, so_hibrido=False, so_lexical=True)
    hib = _melhor(tab, nivel_selecao, so_hibrido=True)
    chave_l = (lex["normalizacao"], int(lex["min_df"]), "{n}", "lexical")
    chave_h = (hib["normalizacao"], int(hib["min_df"]), "{n}", hib["config"])
    por_nivel = {}
    for nivel in niveis:
        if nivel not in cal_por_nivel:
            continue
        teste = ~cal_por_nivel[nivel]
        pl = posicoes_out[(chave_l[0], chave_l[1], nivel, chave_l[3])][teste]
        ph = posicoes_out[(chave_h[0], chave_h[1], nivel, chave_h[3])][teste]
        ml, mh = metricas_de_posicoes(pl), metricas_de_posicoes(ph)
        d1 = (ph == 1).astype(float) - (pl == 1).astype(float)
        dm = 1.0 / ph - 1.0 / pl
        por_nivel[nivel] = {
            "lexical": ml,
            "hibrido": mh,
            "delta_hit@1": float(d1.mean()), "ic95_hit@1": _bootstrap(d1, n_boot),
            "delta_mrr": float(dm.mean()), "ic95_mrr": _bootstrap(dm, n_boot),
        }
    ref = por_nivel[nivel_selecao]
    adotar = ref["delta_hit@1"] >= 0 and ref["delta_mrr"] >= 0
    return {"lexical_escolhido": lex.to_dict(), "hibrido_escolhido": hib.to_dict(),
            "por_nivel": por_nivel, "adotar_hibrido": bool(adotar)}


def calibrar(df, fabrica_semantico, niveis=("Intermediário", "Avançado"), nivel_selecao="Avançado",
             n_boot=2000, **grade) -> tuple[pd.DataFrame, dict]:
    niveis = list(niveis)
    posicoes_out, cal = calcular_posicoes(df, fabrica_semantico, niveis, **grade)
    tab = tabela(posicoes_out, cal)
    return tab, comparar_no_teste(posicoes_out, cal, tab, niveis, nivel_selecao, n_boot)
