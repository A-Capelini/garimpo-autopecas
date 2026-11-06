#!/usr/bin/env python
"""Compara lexical x semântica (por modelo) x híbrida RRF nos 3 níveis.

Requer Ollama rodando com os modelos baixados (`ollama pull all-minilm bge-m3`).
Uso: python scripts/comparar_embeddings.py [--db data/garimpo.db] [--modelos all-minilm bge-m3]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avaliar import NIVEIS, carregar_df  # mesmo diretório

from garimpo.avaliacao import avaliar, construir_gabarito
from garimpo.recuperacao.embeddings import OllamaEmbedder
from garimpo.recuperacao.lexical import BuscaLexical
from garimpo.recuperacao.semantica import BuscaSemantica, fusao_rrf

PROFUNDIDADE_FUSAO = 50


def linha_tabela(nome: str, m: dict) -> str:
    return f"  {nome:<26}{m['hit@1']:>8.1%}{m['hit@5']:>8.1%}{m['hit@10']:>8.1%}{m['mrr']:>8.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/garimpo.db")
    parser.add_argument("--cache", default="data/embeddings_cache.db")
    parser.add_argument("--modelos", nargs="+", default=["all-minilm", "bge-m3"])
    parser.add_argument(
        "--campo", default="texto_busca", choices=["texto_busca", "descricao"],
        help="coluna usada pra gerar o embedding (a busca lexical usa sempre texto_busca)",
    )
    args = parser.parse_args()

    df = carregar_df(args.db)
    conn_cache = sqlite3.connect(args.cache)

    matrizes = {}
    for modelo in args.modelos:
        inicio = time.perf_counter()
        matrizes[modelo] = OllamaEmbedder(modelo=modelo, conn_cache=conn_cache).embed_lote(
            df[args.campo].tolist()
        )
        print(f"{modelo} ({args.campo}): {matrizes[modelo].shape} em {time.perf_counter() - inicio:.0f}s")

    for nivel, fornecedores in NIVEIS.items():
        mascara = df["fornecedor"].isin(fornecedores).to_numpy()
        subset = df[mascara].reset_index(drop=True)
        gabarito = construir_gabarito(subset)
        print(f"\n{nivel} ({len(subset)} itens, {len(gabarito)} consultas)")
        if not gabarito:
            print("  sem gabarito por GTIN")
            continue

        codigos = subset["codigo"].tolist()
        posicao = {c: i for i, c in enumerate(codigos)}

        lexical = BuscaLexical()
        lexical.indexar(codigos, subset["texto_busca"].tolist())

        def top_lexical(c: str, k: int) -> list[str]:
            res = lexical.buscar(subset.at[posicao[c], "texto_busca"], k=k + 1)
            return [r for r in res if r != c][:k]

        print(f"  {'Método':<26}{'Hit@1':>8}{'Hit@5':>8}{'Hit@10':>8}{'MRR':>8}")
        print(linha_tabela("lexical (TF-IDF)", avaliar(gabarito, lambda c: top_lexical(c, 10))))

        for modelo, matriz_total in matrizes.items():
            matriz = matriz_total[mascara]
            semantica = BuscaSemantica()
            semantica.indexar(codigos, matriz)

            def top_semantica(c: str, k: int) -> list[str]:
                res = semantica.buscar_vetor(matriz[posicao[c]], k=k + 1)
                return [r for r in res if r != c][:k]

            def top_hibrida(c: str, peso_lexical: float) -> list[str]:
                return fusao_rrf(
                    [top_lexical(c, PROFUNDIDADE_FUSAO), top_semantica(c, PROFUNDIDADE_FUSAO)],
                    pesos=[peso_lexical, 1.0],
                )[:10]

            def top_rerank(c: str) -> list[str]:
                return semantica.reranquear(matriz[posicao[c]], top_lexical(c, PROFUNDIDADE_FUSAO))

            print(linha_tabela(f"semântica {modelo}", avaliar(gabarito, lambda c: top_semantica(c, 10))))
            for peso in (1.0, 2.0, 3.0):
                m = avaliar(gabarito, lambda c: top_hibrida(c, peso))
                print(linha_tabela(f"RRF {peso:g}:1 {modelo}", m))
            print(linha_tabela(f"rerank top-50 {modelo}", avaliar(gabarito, top_rerank)))


if __name__ == "__main__":
    main()
