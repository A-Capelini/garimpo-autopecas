"""Passo 2 - avalia um recuperador no gabarito, por nivel de escala.

Exemplos:
  python scripts/02_avaliar.py --retriever lexical
  python scripts/02_avaliar.py --retriever ollama --modelo all-minilm
  python scripts/02_avaliar.py --retriever ollama --modelo bge-m3
Os resultados sao acrescentados em resultados/avaliacao.csv
"""
import argparse
import time

import numpy as np
import pandas as pd

from autopecas.config import NIVEIS, RESULTADOS
from autopecas.data.levels import mascara_nivel
from autopecas.data.loader import carregar_catalogo
from autopecas.evaluation.gold import construir_gold
from autopecas.evaluation.metrics import avaliar
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever


def criar_retriever(args):
    if args.retriever == "lexical":
        return TfidfRetriever()
    if not args.modelo:
        raise SystemExit("Informe --modelo (ex.: --modelo bge-m3)")
    return OllamaEmbeddingRetriever(
        args.modelo, prefixo_documento=args.prefixo_doc, prefixo_consulta=args.prefixo_consulta
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", choices=["lexical", "ollama"], required=True)
    ap.add_argument("--modelo", help="nome do modelo no Ollama (ex.: bge-m3, all-minilm)")
    ap.add_argument("--prefixo-doc", default="", help="prefixo dos documentos (modelos e5/nomic)")
    ap.add_argument("--prefixo-consulta", default="", help="prefixo das consultas (modelos e5/nomic)")
    args = ap.parse_args()

    df = carregar_catalogo()
    retriever = criar_retriever(args)

    t0 = time.time()
    try:
        retriever.fit(df["texto_busca"].tolist())
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")
    t_fit = time.time() - t0

    linhas = []
    for nivel in NIVEIS:
        mascara = mascara_nivel(df, nivel)
        gold = construir_gold(df, mascara)
        t1 = time.time()
        scores = retriever.score_rows(gold.consultas) if len(gold) else np.empty((0, len(df)))
        m = avaliar(scores, gold, mascara)
        m.update(
            retriever=retriever.nome,
            nivel=nivel,
            n_docs=int(mascara.sum()),
            seg_indexacao=round(t_fit, 1),
            ms_por_consulta=round(1000 * (time.time() - t1) / max(len(gold), 1), 2),
        )
        linhas.append(m)

    tabela = pd.DataFrame(linhas)[
        ["retriever", "nivel", "n_docs", "n_consultas", "hit@1", "hit@5", "hit@10", "mrr", "seg_indexacao", "ms_por_consulta"]
    ]
    print(tabela.round(3).to_string(index=False))

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / "avaliacao.csv"
    tabela.to_csv(destino, mode="a", header=not destino.exists(), index=False)
    print(f"\nResultados acrescentados em {destino}")


if __name__ == "__main__":
    main()
