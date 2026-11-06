"""Carga e normalização do catálogo consolidado de fornecedores.

A planilha de origem não tem cabeçalho e tem 6 colunas, nesta ordem:
codigo interno, GTIN, descricao, preco, NCM, fornecedor (letra A-L, N).
Não há quantidade em estoque: é catálogo de oferta, não saldo físico.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

COLUNAS = ["codigo", "gtin", "descricao", "preco", "ncm", "fornecedor"]

_RUIDO_RE = re.compile(r"[^a-z0-9 ]+")
_ESPACOS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Peca:
    codigo: str
    gtin: str | None
    descricao: str
    preco: int
    ncm: str
    fornecedor: str
    texto_busca: str


def carregar_catalogo(caminho_xlsx: str, aba: str = "CONSOLIDADO") -> pd.DataFrame:
    """Lê a planilha bruta e devolve DataFrame com as 6 colunas nomeadas."""
    df = pd.read_excel(caminho_xlsx, sheet_name=aba, header=None, names=COLUNAS, dtype=str)
    df["codigo"] = df["codigo"].str.strip()
    df["descricao"] = df["descricao"].str.strip()
    df["ncm"] = df["ncm"].str.strip()
    df["fornecedor"] = df["fornecedor"].str.strip()
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce").astype("Int64")
    df["gtin"] = df["gtin"].apply(_normalizar_gtin_bruto)
    df["texto_busca"] = df["descricao"].apply(normalizar_texto)
    return df


def _normalizar_gtin_bruto(valor: object) -> str | None:
    """'SEM GTIN' e vazio viram None; resto fica como veio (pode ter sufixo -X)."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto == "" or texto.upper() == "SEM GTIN":
        return None
    return texto


def normalizar_texto(texto: str) -> str:
    """minúsculas, sem acento, sem pontuação — usado no índice de busca."""
    if texto is None:
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = _RUIDO_RE.sub(" ", texto)
    texto = _ESPACOS_RE.sub(" ", texto).strip()
    return texto


def gtin_valido(gtin: str | None) -> bool:
    """Valida dígito verificador GS1. Aceita GTIN-8/12/13/14 só-dígitos."""
    if not isinstance(gtin, str) or not gtin.isdigit() or len(gtin) not in (8, 12, 13, 14):
        return False
    digitos = [int(c) for c in gtin]
    corpo, esperado = digitos[:-1], digitos[-1]
    corpo_14 = [0] * (13 - len(corpo)) + corpo  # alinha à direita em 13 posições
    soma = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(corpo_14)))
    checado = (10 - soma % 10) % 10
    return checado == esperado
