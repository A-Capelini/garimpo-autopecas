"""Metricas de recuperacao: Hit@k, Recall@k, MRR e MRR@10."""
import numpy as np

from autopecas.evaluation.gold import Gold

DECIMAIS_EMPATE = 10


def _ordenar(scores: np.ndarray, gold: Gold, mascara: np.ndarray | None) -> np.ndarray:
    # Arredonda para que empates reais (descricoes repetidas -> vetores identicos) nao sejam
    # desfeitos por ruido de ponto flutuante (~1e-16); o desempate passa a ser a posicao
    # (argsort estavel), igual para qualquer formato de matriz.
    s = np.round(scores.astype(np.float64), DECIMAIS_EMPATE)
    if mascara is not None:
        s[:, ~mascara] = -np.inf
    s[np.arange(s.shape[0]), gold.consultas] = -np.inf
    return np.argsort(-s, axis=1, kind="stable")


def primeira_posicao(scores: np.ndarray, gold: Gold, mascara: np.ndarray | None = None) -> np.ndarray:
    """Posicao (1-based) do primeiro equivalente no ranking de cada consulta."""
    ordem = _ordenar(scores, gold, mascara)
    return np.array(
        [np.flatnonzero(np.isin(ordem[i], gold.relevantes[i])).min() + 1 for i in range(len(ordem))],
        dtype=int,
    )


def metricas_de_posicoes(pos: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Hit@k, MRR e MRR@10 a partir das posicoes do primeiro equivalente."""
    if len(pos) == 0:
        return {"n_consultas": 0, "mrr": float("nan"), "mrr@10": float("nan"),
                **{f"hit@{k}": float("nan") for k in ks}}
    r = {
        "n_consultas": int(len(pos)),
        "mrr": float(np.mean(1.0 / pos)),
        "mrr@10": float(np.mean(np.where(pos <= 10, 1.0 / pos, 0.0))),
    }
    for k in ks:
        r[f"hit@{k}"] = float(np.mean(pos <= k))
    return r


def avaliar(
    scores: np.ndarray,
    gold: Gold,
    mascara: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Avalia uma matriz de scores (n_consultas x n_documentos).

    - Linha i de `scores` corresponde a gold.consultas[i].
    - A propria consulta e excluida do ranking.
    - `mascara` (por documento) restringe o corpus (ex.: um nivel de escala).

    Hit@k    : fracao de consultas com ao menos um equivalente no top-k.
    Recall@k : media da fracao dos equivalentes encontrados no top-k.
    MRR      : media de 1 / (posicao do primeiro equivalente).
    MRR@10   : igual, mas vale 0 se o primeiro equivalente passa da posicao 10
               (definicao usada no pipeline do Vitor e nos slides).
    """
    n_q = scores.shape[0]
    if n_q == 0:
        return {**metricas_de_posicoes(np.array([], dtype=int), ks),
                **{f"recall@{k}": float("nan") for k in ks}}

    ordem = _ordenar(scores, gold, mascara)
    primeiro = np.empty(n_q, dtype=int)
    recall = {k: np.empty(n_q) for k in ks}
    for i in range(n_q):
        rel = gold.relevantes[i]
        achou = np.flatnonzero(np.isin(ordem[i], rel))
        primeiro[i] = achou.min() + 1
        for k in ks:
            recall[k][i] = np.isin(ordem[i, :k], rel).sum() / len(rel)

    resultado = metricas_de_posicoes(primeiro, ks)
    for k in ks:
        resultado[f"recall@{k}"] = float(np.mean(recall[k]))
    return resultado
