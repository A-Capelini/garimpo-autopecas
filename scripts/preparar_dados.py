#!/usr/bin/env python
"""Carrega a planilha, valida, grava no SQLite e imprime relatório de qualidade.

Uso: python scripts/preparar_dados.py <caminho.xlsx> [--db data/garimpo.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from garimpo.banco import conectar, gravar_catalogo
from garimpo.dados import carregar_catalogo, gtin_valido


def relatorio_qualidade(df) -> str:
    total = len(df)
    gtin_col = df["gtin"]
    validos = gtin_col.apply(gtin_valido).sum()
    vazios = gtin_col.isna().sum()
    igual_codigo = (gtin_col == df["codigo"]).sum()
    sem_gtin_textual = total - validos - vazios - igual_codigo

    descricoes_repetidas = df["descricao"].duplicated(keep=False).sum()
    ncm_generico = (df["ncm"] == "87089990").sum()

    linhas = [
        f"Total de linhas: {total}",
        f"GTIN válido (GS1): {validos} ({validos / total:.1%})",
        f"GTIN vazio: {vazios} ({vazios / total:.1%})",
        f"GTIN repete código interno: {igual_codigo} ({igual_codigo / total:.1%})",
        f"GTIN 'SEM GTIN' ou inválido: {sem_gtin_textual} ({sem_gtin_textual / total:.1%})",
        f"Descrições repetidas (linhas envolvidas): {descricoes_repetidas}",
        f"NCM genérico 87089990: {ncm_generico} ({ncm_generico / total:.1%})",
        "",
        "Fornecedores: " + ", ".join(sorted(df["fornecedor"].unique())),
    ]
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", help="planilha consolidada (aba CONSOLIDADO)")
    parser.add_argument("--db", default="data/garimpo.db")
    args = parser.parse_args()

    df = carregar_catalogo(args.xlsx)
    print(relatorio_qualidade(df))

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = conectar(args.db)
    gravar_catalogo(conn, df)
    conn.close()
    print(f"\nGravado em {args.db}")


if __name__ == "__main__":
    main()
