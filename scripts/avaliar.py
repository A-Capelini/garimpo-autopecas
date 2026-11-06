#!/usr/bin/env python
"""Avalia o baseline lexical nos 3 níveis de escala (Tabela 5 do relato).

Uso: python scripts/avaliar.py [--db data/garimpo.db]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from garimpo.avaliacao import avaliar, construir_gabarito
from garimpo.recuperacao.lexical import BuscaLexical

NIVEIS = {
    "Básico": set("ABCD"),
    "Intermediário": set("ABCDEFGHI"),
    "Avançado": set("ABCDEFGHIJKLN"),
}


def carregar_df(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql("select * from pecas", conn)
    conn.close()
    return df


def avaliar_nivel(df: pd.DataFrame, fornecedores: set[str]) -> dict[str, float]:
    subset = df[df["fornecedor"].isin(fornecedores)].reset_index(drop=True)
    gabarito = construir_gabarito(subset)

    indice = BuscaLexical()
    indice.indexar(subset["codigo"].tolist(), subset["texto_busca"].tolist())

    def buscar(codigo_consulta: str) -> list[str]:
        texto = subset.loc[subset["codigo"] == codigo_consulta, "texto_busca"].iloc[0]
        resultados = indice.buscar(texto, k=11)
        return [r for r in resultados if r != codigo_consulta][:10]

    resultado = avaliar(gabarito, buscar)
    resultado["itens_no_corpus"] = len(subset)
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/garimpo.db")
    args = parser.parse_args()

    df = carregar_df(args.db)

    print(f"{'Nível':<15}{'Itens':>8}{'Consultas':>11}{'Hit@1':>9}{'Hit@5':>9}{'Hit@10':>9}{'MRR':>8}")
    for nome, fornecedores in NIVEIS.items():
        m = avaliar_nivel(df, fornecedores)
        if m["n_consultas"] == 0:
            print(f"{nome:<15}{m['itens_no_corpus']:>8}{'sem gabarito':>11}")
            continue
        print(
            f"{nome:<15}{m['itens_no_corpus']:>8}{m['n_consultas']:>11}"
            f"{m['hit@1']:>9.1%}{m['hit@5']:>9.1%}{m['hit@10']:>9.1%}{m['mrr']:>8.3f}"
        )


if __name__ == "__main__":
    main()
