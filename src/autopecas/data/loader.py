"""Leitura da planilha bruta e do catalogo processado."""
from pathlib import Path

import pandas as pd

from autopecas.config import ARQUIVO_XLSX, CATALOGO_CSV, COLUNAS

_COLUNAS_TEXTO = ("codigo_interno", "gtin", "descricao", "ncm", "fornecedor")


def carregar_xlsx(caminho: Path | str = ARQUIVO_XLSX) -> pd.DataFrame:
    """Le a planilha bruta (sem cabecalho) e devolve um DataFrame com nomes de colunas.

    Codigos e GTIN ficam como TEXTO (preserva zeros a esquerda, ex.: '01667').
    """
    df = pd.read_excel(
        caminho,
        header=None,
        names=COLUNAS,
        usecols=list(range(len(COLUNAS))),  # a aba declara 26 colunas, so 6 tem dados
        dtype=str,
    )
    for col in _COLUNAS_TEXTO:
        df[col] = df[col].str.strip()
    df["fornecedor"] = df["fornecedor"].str.upper()
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    return df.reset_index(drop=True)


def carregar_catalogo(caminho: Path | str = CATALOGO_CSV) -> pd.DataFrame:
    """Le o catalogo ja processado (saida do script 01)."""
    df = pd.read_csv(
        caminho,
        dtype={c: str for c in _COLUNAS_TEXTO + ("gtin_status", "gtin_chave", "texto_busca")},
        keep_default_na=False,
        na_values=[""],
    )
    return df.reset_index(drop=True)
