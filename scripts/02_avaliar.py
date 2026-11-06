"""Passo 2 - avalia um recuperador no gabarito, por nivel de escala.

Dois gabaritos:
  - por GTIN (padrao): consulta = item do catalogo; niveis Intermediario e Avancado (o Basico tem 0);
  - manual (--gabarito): perguntas em texto livre escritas pelo grupo (docs/GABARITO_BASICO.md).

Exemplos (gabarito por GTIN):
  python scripts/02_avaliar.py --retriever lexical
  python scripts/02_avaliar.py --retriever ollama --modelo bge-m3
  python scripts/02_avaliar.py --retriever hibrido --modelo bge-m3 --metodo rrf --peso-lexical 3 --peso-semantico 1
  python scripts/02_avaliar.py --retriever hibrido --modelo bge-m3 --preset escolhido
Os resultados sao acrescentados em resultados/avaliacao.csv (o nome do retriever traz os parametros).
lexical/hibrido: indice lexical reajustado em cada nivel; ollama: catalogo indexado uma vez e fatiado.
O hibrido usa os embeddings do cache SQLite: rode antes '--retriever ollama' para aquecer o cache.

Exemplos (gabarito manual, nivel Basico por padrao):
  python scripts/02_avaliar.py --gabarito data/raw/gabarito_basico.csv --so-validar
  python scripts/02_avaliar.py --gabarito data/raw/gabarito_basico.csv --retriever todos
  python scripts/02_avaliar.py --gabarito data/raw/gabarito_basico.csv --retriever lexical
Acrescenta em resultados/avaliacao_texto.csv e grava resultados/revisao_<nivel>_<retriever>.md.
'--retriever todos' (so com --gabarito) mede lexical, bge-m3 e hibrido escolhido na mesma execucao
e imprime a diferenca pareada hibrido - lexical.
"""
import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

from autopecas.config import CATALOGO_CSV, NIVEIS, RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.data.quality import normalizar_texto
from autopecas.evaluation.gabarito_texto import (
    ErroGabarito,
    avisos_do_gabarito,
    ler_gabarito,
    resumo_do_gabarito,
    validar_contra_catalogo,
)
from autopecas.evaluation.runner import avaliar_niveis
from autopecas.evaluation.runner_texto import (
    acrescentar_csv,
    avaliar_perguntas,
    comparar_pareado,
    metricas_por_tipo,
    posicoes_por_pergunta,
    relatorio_revisao,
    tabela_de,
    tabela_legivel,
)
from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever

# Configuracao adotada na Etapa 1 (decisoes 21 a 23)
PRESET_ESCOLHIDO = dict(
    normalizacao="simples", min_df=1, sem_sublinear=False,
    metodo="rrf", peso_lexical=3.0, peso_semantico=1.0, k_rrf=1, profundidade=50,
)


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


def _aplicar_preset(args) -> None:
    if args.preset == "escolhido":
        for k, v in PRESET_ESCOLHIDO.items():
            setattr(args, k.replace("-", "_"), v)


def _slug(texto: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "-", ascii_).strip("-").lower()


def _validar_gabarito(args, df):
    """Le e confere o gabarito. Imprime avisos e resumo; encerra com mensagem clara se houver erros."""
    try:
        perguntas = ler_gabarito(args.gabarito)
        erros = validar_contra_catalogo(perguntas, df, args.nivel)
        if erros:
            raise ErroGabarito(erros)
    except ErroGabarito as e:
        raise SystemExit(f"\nGabarito inválido ({args.gabarito}):\n{e}")
    r = resumo_do_gabarito(perguntas)
    print(f"Gabarito OK: {r['perguntas']} perguntas ({r['com_resposta']} com resposta, "
          f"{r['sem_resposta']} sem resposta), {r['pecas_distintas']} peças distintas.")
    print(f"  por tipo: {r['por_tipo']}")
    print(f"  por autor: {r['por_autor']}")
    for aviso in avisos_do_gabarito(perguntas):
        print(f"AVISO: {aviso}")
    return perguntas


def _rodar_gabarito(args, df) -> None:
    perguntas = _validar_gabarito(args, df)
    if args.so_validar:
        return
    manter = args.normalizacao == "oficial"
    normalizar = (lambda t: t.strip()) if args.campo == "descricao" else (
        lambda t: normalizar_texto(t, manter_simbolos=manter))

    if args.retriever == "todos":
        base = argparse.Namespace(**vars(args))
        base.modelo = base.modelo or "bge-m3"
        base.preset = "escolhido"  # 'todos' mede sempre a configuracao das decisoes 21 a 23
        _aplicar_preset(base)
        configs = []
        for nome in ("lexical", "ollama", "hibrido"):
            a = argparse.Namespace(**vars(base))
            a.retriever = nome
            configs.append(a)
    else:
        configs = [args]

    resultados = []
    try:
        for a in configs:
            print(f"\n>> {a.retriever}: indexando o nível {args.nivel} e medindo...", file=sys.stderr)
            resultados.append(avaliar_perguntas(
                df, perguntas, criar_fabrica(a), nivel=args.nivel,
                coluna_texto=args.campo, normalizar=normalizar,
            ))
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")

    tabela = tabela_de(resultados)
    legivel = tabela_legivel(tabela)
    print(f"\nNível {args.nivel}: {tabela['n_docs'].iloc[0]} itens; "
          f"{tabela['n_consultas'].iloc[0]} perguntas medidas (IC95% por bootstrap sobre as perguntas)\n")
    print(legivel.to_string(index=False))
    if len(resultados) == 3:
        lex, _, hib = resultados
        c = comparar_pareado(hib.posicoes, lex.posicoes)
        print("\nDiferença pareada híbrido − léxico (mesmas perguntas):")
        for m in ("hit@1", "hit@5", "mrr"):
            lo, hi = c[f"ic95_{m}"]
            print(f"  Δ {m}: {c[f'delta_{m}']:+.3f}  IC95% [{lo:+.3f}; {hi:+.3f}]")
        print("  (intervalo que contém 0 = diferença não conclusiva com este número de perguntas)")

    por_tipo = metricas_por_tipo(resultados)
    print("\nPor tipo de pergunta (descritivo: poucas perguntas por tipo, sem intervalo):")
    for nome, g in por_tipo.groupby("retriever", sort=False):
        print(f"  {nome}")
        print(g.drop(columns=["retriever", "nivel"]).round(3).to_string(index=False).replace("\n", "\n  "))
    posicoes = posicoes_por_pergunta(resultados)
    colunas = [r.linha["retriever"] for r in resultados]
    ruins = posicoes[(posicoes[colunas] > 1).any(axis=1)]
    print(f"\nPerguntas em que algum recuperador não acertou em 1º lugar ({len(ruins)} de {len(posicoes)}); "
          f"posição do 1º equivalente ({args.nivel} tem {tabela['n_docs'].iloc[0]} itens):")
    print(ruins.to_string(index=False))

    # Rastreabilidade: cada linha do CSV aponta para a versao exata do gabarito que a gerou.
    sha = hashlib.sha256(args.gabarito.read_bytes()).hexdigest()[:12]
    print(f"\nGabarito: {args.gabarito} (sha256 {sha}…)")
    tabela.insert(1, "normalizacao", args.normalizacao)
    tabela.insert(2, "campo", args.campo)
    tabela.insert(3, "gabarito_sha256", sha)
    destino = args.resultados / "avaliacao_texto.csv"
    try:
        acrescentar_csv(tabela, destino)
    except ValueError as e:
        raise SystemExit(f"\nERRO: {e}")
    print(f"\nResultados acrescentados em {destino}")
    por_tipo.insert(2, "gabarito_sha256", sha)
    try:
        acrescentar_csv(por_tipo, args.resultados / "avaliacao_texto_por_tipo.csv")
    except ValueError as e:
        raise SystemExit(f"\nERRO: {e}")
    posicoes.to_csv(args.resultados / f"posicoes_{_slug(args.nivel)}.csv", index=False)  # so a ultima medicao
    for res in resultados:
        caminho = args.resultados / f"revisao_{_slug(args.nivel)}_{_slug(res.linha['retriever'])}.md"
        caminho.write_text(relatorio_revisao(res, perguntas), encoding="utf-8")
        print(f"Revisão manual: {caminho}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", choices=["lexical", "ollama", "hibrido", "todos"], required=True)
    ap.add_argument("--modelo", help="nome do modelo no Ollama (ex.: bge-m3)")
    ap.add_argument("--prefixo-doc", default="", help="prefixo dos documentos (modelos e5/nomic)")
    ap.add_argument("--prefixo-consulta", default="", help="prefixo das consultas (modelos e5/nomic)")
    ap.add_argument("--campo", default="texto_busca", choices=["texto_busca", "descricao"],
                    help="coluna indexada (padrao: texto_busca)")
    ap.add_argument("--min-df", type=int, default=1, help="min_df do TF-IDF (padrao: 1)")
    ap.add_argument("--sem-sublinear", action="store_true", help="desliga sublinear_tf (padrao: ligado)")
    ap.add_argument("--normalizacao", choices=["oficial", "simples"], default="simples",
                    help="simples (padrao, decisao da Etapa 1): todo simbolo vira espaco; "
                         "oficial: mantem '/' e '>' (padrao anterior)")
    ap.add_argument("--metodo", choices=["rrf", "soma"], default="rrf", help="fusao do hibrido")
    ap.add_argument("--peso-lexical", type=float, default=None, help="rrf: padrao 3; soma: [0,1], padrao 0.5")
    ap.add_argument("--peso-semantico", type=float, default=1.0, help="so rrf")
    ap.add_argument("--k-rrf", type=int, default=60, help="so rrf (a configuracao escolhida usa 1)")
    ap.add_argument("--profundidade", type=int, default=50, help="so rrf: top-N de cada fonte")
    ap.add_argument("--preset", choices=["escolhido"],
                    help="'escolhido' = configuracao das decisoes 21 a 23 (texto simples, min_df=1, "
                         "RRF 3:1, k=1, profundidade 50); sobrescreve os parametros acima")
    ap.add_argument("--gabarito", type=Path,
                    help="CSV com o gabarito manual (perguntas em texto livre); ver docs/GABARITO_BASICO.md")
    ap.add_argument("--nivel", choices=list(NIVEIS), default="Básico",
                    help="nivel avaliado com --gabarito (padrao: Básico)")
    ap.add_argument("--so-validar", action="store_true", help="com --gabarito: so confere o arquivo e sai")
    ap.add_argument("--catalogo", type=Path, default=CATALOGO_CSV,
                    help="catalogo processado (saida do script 01); padrao: data/processed/catalogo.csv")
    ap.add_argument("--resultados", type=Path, default=RESULTADOS, help="pasta de saida (padrao: resultados/)")
    args = ap.parse_args()

    if args.retriever == "todos" and not args.gabarito:
        raise SystemExit("--retriever todos só existe com --gabarito")
    if args.so_validar and not args.gabarito:
        raise SystemExit("--so-validar só existe com --gabarito")
    _aplicar_preset(args)

    df = carregar_catalogo(args.catalogo)
    # Sempre recalculado a partir da descricao: a avaliacao nao depende do catalogo.csv.
    manter = args.normalizacao == "oficial"
    df["texto_busca"] = df["descricao"].map(lambda t: normalizar_texto(t, manter_simbolos=manter))

    if args.gabarito:
        _rodar_gabarito(args, df)
        return

    try:
        tabela = avaliar_niveis(df, criar_fabrica(args), coluna_texto=args.campo)
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")
    tabela.insert(1, "normalizacao", args.normalizacao)
    tabela.insert(2, "campo", args.campo)
    print(tabela.round(3).to_string(index=False))

    args.resultados.mkdir(parents=True, exist_ok=True)
    destino = args.resultados / "avaliacao.csv"
    tabela.to_csv(destino, mode="a", header=not destino.exists(), index=False)
    print(f"\nResultados acrescentados em {destino}")


if __name__ == "__main__":
    main()
