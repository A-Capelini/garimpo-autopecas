"""Divisao calibracao/teste das consultas, por GTIN (hash estavel).

Todas as consultas de um mesmo GTIN caem do mesmo lado: um produto nunca aparece nos dois
conjuntos. A divisao depende so do GTIN e da semente, nao da ordem das linhas.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

SEMENTE = "garimpo-pi4"


def mascara_calibracao(chaves: Sequence[str], frac_calibracao: float = 0.6, semente: str = SEMENTE) -> np.ndarray:
    """True para as chaves (GTINs) que vao para a calibracao; False para o teste."""
    h = np.array(
        [int(hashlib.sha1(f"{semente}|{k}".encode()).hexdigest()[:8], 16) % 10000 for k in chaves]
    )
    return h < int(round(frac_calibracao * 10000))
