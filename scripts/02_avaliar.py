"""Passo 2 - avalia um recuperador no gabarito, por nivel de escala.

Exemplos:
  python scripts/02_avaliar.py --retriever lexical
  python scripts/02_avaliar.py --retriever ollama --modelo bge-m3
  python scripts/02_avaliar.py --retriever hibrido --modelo bge-m3 --metodo rrf --peso-lexical 3 --peso-semantico 1
  python scripts/02_avaliar.py --retriever hibrido --modelo bge-m3 --metodo soma --peso-lexical 0.5
Os resultados sao acrescentados em resultados/avaliacao.csv (o nome do retriever traz os parametros).
lexical/hibrido: indice lexical reajustado em cada nivel; ollama: catalogo indexado uma vez e fatiado.
O hibrido usa os embeddings do cache SQLite: rode antes '--retriever ollama' para aquecer o cache.
"""
import argparse

from autopecas.config import RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.data.quality import normalizar_texto
from autopecas.evaluation.runner import avaliar_niveis
from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever


def criar_fabrica(args):
    lexical = lambda: TfidfRetriever(min_df=args.min_df, sublinear_tf=not args.sem_sublinear)
    if args.retriever == "lexical":
        return lexical
    if not args.modelo:
        raise SystemExit("Informe --modelo (ex.: --modelo bge-m3)")
    semantico = lambda: OllamaEmbeddingRetriever(
        args.modelo, prefixo_documento=args.prefixo_doc, prefixo_consulta=args.prefixo_consulta
    )
    if args.retriever == "ollama":
        return semantico
    peso_lex = args.peso_lexical if args.peso_lexical is not None else (3.0 if args.metodo == "rrf" else 0.5)
    return lambda: HybridRetriever(
        lexical(),
        semantico(),
        metodo=args.metodo,
        peso_lexical=peso_lex,
        peso_semantico=args.peso_semantico,
        k_rrf=args.k_rrf,
        profundidade=args.profundidade,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", choices=["lexical", "ollama", "hibrido"], required=True)
    ap.add_argument("--modelo", help="nome do modelo no Ollama (ex.: bge-m3)")
    ap.add_argument("--prefixo-doc", default="", help="prefixo dos documentos (modelos e5/nomic)")
    ap.add_argument("--prefixo-consulta", default="", help="prefixo das consultas (modelos e5/nomic)")
    ap.add_argument("--campo", default="texto_busca", choices=["texto_busca", "descricao"],
                    help="coluna indexada (padrao: texto_busca)")
    ap.add_argument("--min-df", type=int, default=2, help="min_df do TF-IDF (padrao oficial: 2)")
    ap.add_argument("--sem-sublinear", action="store_true", help="desliga sublinear_tf (padrao oficial: ligado)")
    ap.add_argument("--normalizacao", choices=["oficial", "simples"], default="oficial",
                    help="simples: todo simbolo vira espaco (como no pipeline do Vitor)")
    ap.add_argument("--metodo", choices=["rrf", "soma"], default="rrf", help="fusao do hibrido")
    ap.add_argument("--peso-lexical", type=float, default=None, help="rrf: padrao 3; soma: [0,1], padrao 0.5")
    ap.add_argument("--peso-semantico", type=float, default=1.0, help="so rrf")
    ap.add_argument("--k-rrf", type=int, default=60, help="so rrf")
    ap.add_argument("--profundidade", type=int, default=50, help="so rrf: top-N de cada fonte")
    args = ap.parse_args()

    df = carregar_catalogo()
    if args.normalizacao == "simples":
        df["texto_busca"] = df["descricao"].map(lambda t: normalizar_texto(t, manter_simbolos=False))
    try:
        tabela = avaliar_niveis(df, criar_fabrica(args), coluna_texto=args.campo)
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")
    print(tabela.round(3).to_string(index=False))

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / "avaliacao.csv"
    tabela.to_csv(destino, mode="a", header=not destino.exists(), index=False)
    print(f"\nResultados acrescentados em {destino}")


if __name__ == "__main__":
    main()
