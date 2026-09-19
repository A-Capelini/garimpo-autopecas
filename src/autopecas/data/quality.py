"""Qualidade dos dados: validacao de GTIN e normalizacao do texto de busca."""
import re
import unicodedata

import pandas as pd

STATUS_GTIN = (
    "valido",
    "igual_codigo_interno",
    "sem_gtin_texto",
    "ausente",
    "invalido",
)


def digito_verificador_ok(gtin: str) -> bool:
    """Valida o digito verificador GS1 (GTIN-8/12/13/14)."""
    if not gtin.isdigit() or len(gtin) not in (8, 12, 13, 14):
        return False
    corpo, dv = gtin[:-1], int(gtin[-1])
    soma = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(corpo)))
    return (10 - soma % 10) % 10 == dv


def classificar_gtin(gtin, codigo_interno) -> str:
    """Classifica o campo GTIN de uma linha em um dos STATUS_GTIN."""
    if pd.isna(gtin) or str(gtin).strip() == "":
        return "ausente"
    g = str(gtin).strip()
    if g.upper() == "SEM GTIN":
        return "sem_gtin_texto"
    if not pd.isna(codigo_interno) and g == str(codigo_interno).strip():
        return "igual_codigo_interno"
    return "valido" if digito_verificador_ok(g) else "invalido"


def normalizar_texto(texto) -> str:
    """Texto de busca: minusculo, sem acento, sem o sufixo '-:' do fornecedor B."""
    if pd.isna(texto):
        return ""
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s*-:\s*$", "", s)
    s = re.sub(r"[^a-z0-9/>\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def aplicar_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona gtin_status, gtin_chave (so GTIN valido), texto_busca e preco_zero."""
    out = df.copy()
    out["gtin_status"] = [classificar_gtin(g, c) for g, c in zip(out["gtin"], out["codigo_interno"])]
    out["gtin_chave"] = out["gtin"].where(out["gtin_status"] == "valido")
    out["texto_busca"] = out["descricao"].map(normalizar_texto)
    out["preco_zero"] = out["preco"].fillna(0) <= 0
    return out


def resumo_qualidade(df: pd.DataFrame) -> dict:
    """Numeros-chave para o relatorio de diagnostico."""
    desc = df["descricao"].str.upper().str.strip()
    return {
        "linhas": int(len(df)),
        "fornecedores": sorted(df["fornecedor"].unique().tolist()),
        "linhas_por_fornecedor": df["fornecedor"].value_counts().sort_index().to_dict(),
        "gtin_status": df["gtin_status"].value_counts().to_dict(),
        "ncm_distintos": int(df["ncm"].nunique()),
        "descricoes_repetidas": int(desc.duplicated(keep=False).sum()),
        "preco_zero": int(df["preco_zero"].sum()),
    }
