"""Passo 10 - medicao da Etapa 4: ordenacao por veiculo (base x +veiculo) e diagnostico da hipotese.

Mede, no Basico e no gabarito manual, cada recuperador SEM e COM a regra de `retrieval/veiculo.py`
(os dois caminhos usam a mesma funcao; o app faz o mesmo), compara PAREADO (+veiculo - base) e classifica
o que fica a frente da 1a peca esperada (hipotese do veiculo, CONTEXTO 5.5).

Uso:
  python scripts/10_medir_veiculo.py                          # lexical, bge-m3 e hibrido (precisa do Ollama)
  python scripts/10_medir_veiculo.py --so-lexical             # sem Ollama
  python scripts/10_medir_veiculo.py --gabarito outro.csv

Configuracao medida: a escolhida nas decisoes 21 a 23 (texto simples, min_df=1, RRF 3:1, k_rrf=1, prof. 50).
Saidas em resultados/ (fora do Git): avaliacao_texto.csv e avaliacao_texto_por_tipo.csv (linhas novas,
retriever '...+veiculo'), posicoes_basico_veiculo.csv, diagnostico_veiculo.csv, revisao_basico_*+veiculo.md.
"""
import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

from autopecas.config import CATALOGO_CSV, NIVEIS, RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA, normalizar_texto
from autopecas.evaluation.diagnostico_veiculo import itens_a_frente, resumo
from autopecas.evaluation.gabarito_texto import (
    ErroGabarito,
    avisos_do_gabarito,
    ler_gabarito,
    validar_contra_catalogo,
)
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
from autopecas.retrieval.veiculo import ComVeiculo


def _slug(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _fabricas(so_lexical: bool) -> dict:
    lexical = lambda: TfidfRetriever(min_df=1, sublinear_tf=True)
    fabricas = {"lexical": lexical}
    if not so_lexical:
        fabricas["bge-m3"] = lambda: OllamaEmbeddingRetriever("bge-m3")
        fabricas["hibrido"] = lambda: HybridRetriever(
            lexical(), OllamaEmbeddingRetriever("bge-m3"), metodo="rrf",
            peso_lexical=3.0, peso_semantico=1.0, k_rrf=1, profundidade=50,
        )
    return fabricas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gabarito", type=Path, default=Path("data/raw/gabarito_basico.csv"))
    ap.add_argument("--nivel", choices=list(NIVEIS), default="Básico")
    ap.add_argument("--so-lexical", action="store_true", help="nao usa o Ollama (so o lexical, base e +veiculo)")
    ap.add_argument("--catalogo", type=Path, default=CATALOGO_CSV)
    ap.add_argument("--resultados", type=Path, default=RESULTADOS)
    args = ap.parse_args()

    df = carregar_catalogo(args.catalogo)
    df["texto_busca"] = df["descricao"].map(
        lambda t: normalizar_texto(t, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA))
    try:
        perguntas = ler_gabarito(args.gabarito)
    except (ErroGabarito, OSError) as e:
        raise SystemExit(f"\nGabarito inválido ({args.gabarito}):\n{e}")
    erros = validar_contra_catalogo(perguntas, df, args.nivel)
    if erros:
        raise SystemExit("\nGabarito não bate com o catálogo:\n" + "\n".join(f"- {e}" for e in erros))
    for aviso in avisos_do_gabarito(perguntas):
        print("AVISO:", aviso)
    sha = hashlib.sha256(args.gabarito.read_bytes()).hexdigest()[:12]

    base, com = [], []
    try:
        for nome, fabrica in _fabricas(args.so_lexical).items():
            print(f">> {nome}: medindo sem e com a regra do veículo...", file=sys.stderr)
            base.append(avaliar_perguntas(df, perguntas, fabrica, nivel=args.nivel))
            com.append(avaliar_perguntas(df, perguntas, lambda f=fabrica: ComVeiculo(f()), nivel=args.nivel))
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")
    resultados = [r for par in zip(base, com) for r in par]

    tabela = tabela_de(resultados)
    print(f"\nNível {args.nivel}: {tabela['n_docs'].iloc[0]} itens; {tabela['n_consultas'].iloc[0]} perguntas "
          f"medidas (gabarito sha256 {sha}…; IC95% por bootstrap sobre as perguntas)\n")
    print(tabela_legivel(tabela).to_string(index=False))

    print("\nDiferença pareada (+veiculo − base), mesmas perguntas:")
    for b, c in zip(base, com):
        d = comparar_pareado(c.posicoes, b.posicoes)
        partes = [f"Δ {m}: {d[f'delta_{m}']:+.3f} [{d[f'ic95_{m}'][0]:+.3f}; {d[f'ic95_{m}'][1]:+.3f}]"
                  for m in ("hit@1", "hit@5", "mrr")]
        print(f"  {b.linha['retriever']}: " + " | ".join(partes))
    print("  (intervalo que contém 0 = diferença não conclusiva com este número de perguntas)")

    por_tipo = metricas_por_tipo(resultados)
    print("\nPor tipo de pergunta (descritivo; poucas perguntas por tipo):")
    for nome, g in por_tipo.groupby("retriever", sort=False):
        print(f"  {nome}")
        print(g.drop(columns=["retriever", "nivel"]).round(3).to_string(index=False).replace("\n", "\n  "))

    posicoes = posicoes_por_pergunta(resultados)
    for b, c in zip(base, com):
        nb, nc = b.linha["retriever"], c.linha["retriever"]
        mudou = posicoes[posicoes[nb] != posicoes[nc]][["id", "pergunta", nb, nc]]
        piorou = mudou[mudou[nc] > mudou[nb]]
        print(f"\n{nc}: {len(mudou)} pergunta(s) mudaram de posição; {len(piorou)} piorou(aram).")
        if len(mudou):
            print(mudou.to_string(index=False))

    diag = itens_a_frente(base[0])
    r = resumo(diag)
    print(f"\nDiagnóstico da hipótese do veículo ({base[0].linha['retriever']}, sem a regra): "
          f"{r['perguntas_erradas']} pergunta(s) em que a 1ª peça esperada não é a 1ª.")
    print(f"  Nas {r['perguntas_com_veiculo']} que citam veículo: {r['itens_a_frente']} itens à frente → "
          f"{r['cita_o_veiculo']} citam o veículo da pergunta, {r['sem_veiculo']} sem veículo (genéricos), "
          f"{r['outro_veiculo']} citam só outro veículo ({r['perguntas_com_algum_outro_veiculo']} pergunta(s) com algum).")
    print("  Só a leitura humana de revisao_*.md diz se o item à frente é outra peça ou uma peça válida esquecida.")
    print(diag.to_string(index=False))

    args.resultados.mkdir(parents=True, exist_ok=True)
    tabela.insert(1, "normalizacao", "simples")
    tabela.insert(2, "campo", "texto_busca")
    tabela.insert(3, "gabarito_sha256", sha)
    por_tipo.insert(2, "gabarito_sha256", sha)
    try:
        acrescentar_csv(tabela, args.resultados / "avaliacao_texto.csv")
        acrescentar_csv(por_tipo, args.resultados / "avaliacao_texto_por_tipo.csv")
    except ValueError as e:
        raise SystemExit(f"\nERRO: {e}")
    posicoes.to_csv(args.resultados / "posicoes_basico_veiculo.csv", index=False)
    diag.to_csv(args.resultados / "diagnostico_veiculo.csv", index=False)
    for res in com:
        caminho = args.resultados / f"revisao_{_slug(args.nivel)}_{_slug(res.linha['retriever'])}.md"
        caminho.write_text(relatorio_revisao(res, perguntas), encoding="utf-8")
    print(f"\nResultados acrescentados em {args.resultados}/avaliacao_texto.csv (e _por_tipo); "
          "posicoes_basico_veiculo.csv, diagnostico_veiculo.csv e revisao_*+veiculo.md gravados.")


if __name__ == "__main__":
    main()
