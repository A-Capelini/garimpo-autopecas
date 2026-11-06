from __future__ import annotations

import pandas as pd
import pytest

from garimpo.dados import normalizar_texto


@pytest.fixture
def df_pequeno() -> pd.DataFrame:
    """5 linhas sintéticas: reproduz o caso do relato (3 fornecedores, mesmo GTIN)."""
    linhas = [
        # cabo do capô do Uno, mesmo GTIN em 3 fornecedores (slide 2 do relato)
        {"codigo": "3502", "gtin": "7898110832573", "descricao": "CABO CAPO S/ALAV UNO", "preco": 16, "ncm": "87089990", "fornecedor": "C"},
        {"codigo": "9001", "gtin": "7898110832573", "descricao": "CB CAPO UNO PREMIO S/ALAV", "preco": 20, "ncm": "87089990", "fornecedor": "I"},
        {"codigo": "9002", "gtin": "7898110832573", "descricao": "CB CAPO UNO PRE FIO TDS S/ALAV", "preco": 16, "ncm": "87089990", "fornecedor": "J"},
        # item sem par (GTIN único no catálogo)
        {"codigo": "4001", "gtin": "7891234567895", "descricao": "FILTRO DE OLEO GOL", "preco": 29, "ncm": "84212300", "fornecedor": "A"},
        # sem GTIN
        {"codigo": "5001", "gtin": None, "descricao": "AMORTECEDOR DIANTEIRO FUSCA", "preco": 89, "ncm": "87088000", "fornecedor": "N"},
    ]
    df = pd.DataFrame(linhas)
    df["texto_busca"] = df["descricao"].apply(normalizar_texto)
    return df
