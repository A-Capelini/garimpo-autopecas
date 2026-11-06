"""Medicao do custo de atualizar o catalogo (criterio de aceite da Etapa 2).

Pergunta: adicionar N itens novos exige reindexar o resto? Mede, num catalogo temporario
(nunca o real): importar a planilha menos N itens, indexar, importar os N itens e reindexar.
Os vetores dos textos dos N itens sao removidos do cache antes, para que o custo de embedar
so os novos seja medido de verdade (senao o cache da Etapa 1 os teria).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import OllamaEmbeddingRetriever
from autopecas.store.catalogo import Catalogo


def medir_atualizacao(
    df: pd.DataFrame,
    n_novos: int = 100,
    semente: int = 42,
    cache_origem: Path | str | None = None,
    modelo: str = "bge-m3",
    host: str | None = None,
    sessao=None,
) -> dict:
    """`df`: catalogo com as colunas de `Catalogo.importar` (saida de aplicar_qualidade)."""
    if not 0 < n_novos < len(df):
        raise ValueError("n_novos deve estar entre 1 e len(df) - 1")
    rng = np.random.default_rng(semente)
    sorteados = np.zeros(len(df), dtype=bool)
    sorteados[rng.choice(len(df), size=n_novos, replace=False)] = True
    base, novos = df[~sorteados], df[sorteados]

    with tempfile.TemporaryDirectory(prefix="garimpo_medicao_") as tmp:
        banco = Path(tmp) / "catalogo.db"
        with Catalogo(banco) as cat:
            if cache_origem is not None and Path(cache_origem).exists():
                cat.absorver_cache(cache_origem)
            cat.importar(base)

            emb = OllamaEmbeddingRetriever(modelo, host=host, sessao=sessao, banco=banco)
            lex = TfidfRetriever()

            textos_base = set(base["texto_busca"])
            textos_novos = sorted(set(novos["texto_busca"]) - textos_base)
            removidos = cat.remover_vetores(modelo, [emb.hash_documento(t) for t in textos_novos])

            t0 = time.perf_counter()
            lex.fit(base["texto_busca"].tolist())
            emb.fit(base["texto_busca"].tolist())
            base_s = time.perf_counter() - t0
            chamadas_base = emb.chamadas_http

            vetores_antes = cat.contar_vetores(modelo)
            t1 = time.perf_counter()
            relatorio = cat.importar(novos)
            importar_s = time.perf_counter() - t1

            textos = cat.itens()["texto_busca"].tolist()
            t2 = time.perf_counter()
            lex.fit(textos)
            lexical_s = time.perf_counter() - t2

            t3 = time.perf_counter()
            emb.fit(textos)
            vetores_s = time.perf_counter() - t3
            vetores_calculados = cat.contar_vetores(modelo) - vetores_antes
            chamadas_novos = emb.chamadas_http - chamadas_base
            emb.fechar()

    return {
        "itens_base": len(base),
        "itens_novos": len(novos),
        "textos_novos_distintos": len(textos_novos),
        "vetores_removidos_do_cache": removidos,
        "indexacao_base_s": base_s,
        "chamadas_http_base": chamadas_base,
        "importar_s": importar_s,
        "lexical_s": lexical_s,
        "vetores_s": vetores_s,
        "atualizacao_total_s": importar_s + lexical_s + vetores_s,
        "vetores_calculados": vetores_calculados,
        "chamadas_http_novos": chamadas_novos,
        "relatorio": relatorio.resumo(),
    }


def formatar_markdown(m: dict, n_total: int) -> str:
    linhas = [
        "# Medicao: atualizar o catalogo com itens novos",
        "",
        f"Catalogo base: {m['itens_base']} itens; itens novos: {m['itens_novos']} "
        f"({m['textos_novos_distintos']} textos de busca distintos; "
        f"{m['vetores_removidos_do_cache']} vetores removidos do cache antes da medicao).",
        "",
        "| etapa | tempo (s) |",
        "|---|---:|",
        f"| indexar a base (lexical + vetores do cache) | {m['indexacao_base_s']:.2f} |",
        f"| importar os novos no SQLite | {m['importar_s']:.3f} |",
        f"| reajustar o lexical (TF-IDF) no catalogo de {n_total} | {m['lexical_s']:.2f} |",
        f"| embedar so os textos novos | {m['vetores_s']:.2f} |",
        f"| **atualizacao total** | **{m['atualizacao_total_s']:.2f}** |",
        "",
        f"Vetores calculados na atualizacao: {m['vetores_calculados']} "
        f"({m['chamadas_http_novos']} chamadas HTTP ao Ollama); "
        f"chamadas na indexacao da base: {m['chamadas_http_base']} (0 = tudo veio do cache).",
        "",
        f"Relatorio da importacao: {m['relatorio']}",
        "",
    ]
    return "\n".join(linhas)
