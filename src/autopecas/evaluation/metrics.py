"""Metricas de recuperacao: Hit@k, Recall@k e MRR."""
import numpy as np

from autopecas.evaluation.gold import Gold

DECIMAIS_EMPATE = 10


def avaliar(
    scores: np.ndarray,
    gold: Gold,
    mascara: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Avalia uma matriz de scores (n_consultas x n_documentos)."""
    n_q = scores.shape[0]
    if n_q == 0:
        return {"n_consultas": 0, "mrr": float("nan"), "mrr@10": float("nan"), **{f"hit@{k}": float("nan") for k in ks},
                **{f"recall@{k}": float("nan") for k in ks}}

    # Arredonda para que empates reais (descricoes repetidas -> vetores identicos) nao sejam
    # desfeitos por ruido de ponto flutuante (~1e-16); o desempate passa a ser a posicao
    # (argsort estavel), igual para qualquer formato de matriz.
    s = np.round(scores.astype(np.float64), DECIMAIS_EMPATE)
    if mascara is not None:
        s[:, ~mascara] = -np.inf
    s[np.arange(n_q), gold.consultas] = -np.inf
    ordem = np.argsort(-s, axis=1, kind="stable")

    primeiro = np.empty(n_q, dtype=int)
    recall = {k: np.empty(n_q) for k in ks}
    for i in range(n_q):
        rel = gold.relevantes[i]
        achou = np.flatnonzero(np.isin(ordem[i], rel))
        primeiro[i] = achou.min() + 1
        for k in ks:
            recall[k][i] = np.isin(ordem[i, :k], rel).sum() / len(rel)

    resultado = {
        "n_consultas": int(n_q),
        "mrr": float(np.mean(1.0 / primeiro)),
        "mrr@10": float(np.mean(np.where(primeiro <= 10, 1.0 / primeiro, 0.0))),
    }
    for k in ks:
        resultado[f"hit@{k}"] = float(np.mean(primeiro <= k))
        resultado[f"recall@{k}"] = float(np.mean(recall[k]))
    return resultado
