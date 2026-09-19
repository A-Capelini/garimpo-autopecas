"""Niveis de escala do escopo (Basico -> Intermediario -> Avancado)."""
import numpy as np
import pandas as pd

from autopecas.config import NIVEIS


def fornecedores_do_nivel(df: pd.DataFrame, nivel: str) -> list[str]:
    letras = NIVEIS[nivel]
    presentes = set(df["fornecedor"].unique())
    if letras is None:
        return sorted(presentes)
    return [f for f in letras if f in presentes]


def mascara_nivel(df: pd.DataFrame, nivel: str) -> np.ndarray:
    """Mascara booleana (por posicao) das linhas que pertencem ao nivel."""
    return df["fornecedor"].isin(fornecedores_do_nivel(df, nivel)).to_numpy()
