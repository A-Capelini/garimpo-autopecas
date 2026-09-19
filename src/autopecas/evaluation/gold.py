"""Gabarito (gold set) de equivalencia entre fornecedores, via GTIN valido compartilhado.

Consulta  = uma linha de um fornecedor.
Relevante = linhas de OUTROS fornecedores com o mesmo GTIN valido.
Posicoes sao sempre posicoes (0..n-1) no DataFrame, que deve ter indice padrao.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Gold:
    consultas: np.ndarray  # posicoes das linhas usadas como consulta
    relevantes: tuple  # tuple de np.ndarray (posicoes relevantes de cada consulta)

    def __len__(self) -> int:
        return len(self.consultas)


def construir_gold(df: pd.DataFrame, mascara: np.ndarray | None = None) -> Gold:
    """Constroi o gabarito. `mascara` restringe as linhas consideradas (ex.: um nivel)."""
    if mascara is None:
        mascara = np.ones(len(df), dtype=bool)
    chave = df["gtin_chave"].to_numpy(dtype=object)
    forn = df["fornecedor"].to_numpy(dtype=object)

    grupos: dict[str, list[int]] = {}
    for pos in np.flatnonzero(mascara):
        g = chave[pos]
        if isinstance(g, str):
            grupos.setdefault(g, []).append(int(pos))

    consultas: list[int] = []
    relevantes: list[np.ndarray] = []
    for posicoes in grupos.values():
        if len({forn[p] for p in posicoes}) < 2:
            continue
        for p in posicoes:
            rel = [q for q in posicoes if forn[q] != forn[p]]
            consultas.append(p)
            relevantes.append(np.array(rel, dtype=int))
    return Gold(np.array(consultas, dtype=int), tuple(relevantes))
