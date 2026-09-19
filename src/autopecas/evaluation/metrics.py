"""Metricas de recuperacao: Hit@k, Recall@k e MRR."""
import numpy as np

from autopecas.evaluation.gold import Gold


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
    """
    n_q = scores.shape[0]
    if n_q == 0:
        return {"n_consultas": 0, "mrr": float("nan"), **{f"hit@{k}": float("nan") for k in ks},
                **{f"recall@{k}": float("nan") for k in ks}}

    s = scores.astype(np.float64, copy=True)
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

    resultado = {"n_consultas": int(n_q), "mrr": float(np.mean(1.0 / primeiro))}
    for k in ks:
        resultado[f"hit@{k}"] = float(np.mean(primeiro <= k))
        resultado[f"recall@{k}"] = float(np.mean(recall[k]))
    return resultado
