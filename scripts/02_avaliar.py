"""Passo 2 - avalia um recuperador no gabarito, por nivel de escala.

Exemplos:
  python scripts/02_avaliar.py --retriever lexical
  python scripts/02_avaliar.py --retriever ollama --modelo all-minilm
  python scripts/02_avaliar.py --retriever ollama --modelo bge-m3
Os resultados sao acrescentados em resultados/avaliacao.csv
(lexical: indice reajustado em cada nivel; ollama: catalogo indexado uma vez e fatiado).
"""
import argparse

from autopecas.config import RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.evaluation.runner import avaliar_niveis
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever


def criar_fabrica(args):
    if args.retriever == "lexical":
        return TfidfRetriever
    if not args.modelo:
        raise SystemExit("Informe --modelo (ex.: --modelo bge-m3)")
    return lambda: OllamaEmbeddingRetriever(
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
    try:
        tabela = avaliar_niveis(df, criar_fabrica(args))
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")
    print(tabela.round(3).to_string(index=False))

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / "avaliacao.csv"
    tabela.to_csv(destino, mode="a", header=not destino.exists(), index=False)
    print(f"\nResultados acrescentados em {destino}")


if __name__ == "__main__":
    main()
