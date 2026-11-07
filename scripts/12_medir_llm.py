"""Passo 12 - medicao do LLM local (Etapa 5): tempos, taxa de geracao e guarda, com consultas reais.

Para cada modelo, roda cada consulta pelo caminho real do app (Buscador -> lista -> explicacao em streaming)
e mede: tempo da lista, primeira palavra, resposta completa, `eval rate`, violacoes da guarda e erros. Tambem
mede o custo da regra de veiculo na busca (com e sem), pergunta pendente desde a Etapa 4.

Uso (servidor ligado: OLLAMA_NO_CLOUD=1 ollama serve, em outro terminal; modelos ja baixados):
  python scripts/12_medir_llm.py                                  # gemma3:1b e llama3.2:3b, 12 consultas
  python scripts/12_medir_llm.py --modelos gemma3:1b              # um modelo so
  python scripts/12_medir_llm.py --consultas-arquivo minhas.txt   # uma consulta por linha
  python scripts/12_medir_llm.py --sem-llm                        # so lista e custo da regra de veiculo
  python scripts/12_medir_llm.py --modelos lfm3:1.7b --pensar desligado   # modelo com a marca "thinking"
  python scripts/12_medir_llm.py --prompt b --rotulo promptb      # variante B do prompt (2 exemplos)
  python scripts/12_medir_llm.py --prompt c --rotulo promptc      # variante C (recusa primeiro, 3 itens)

--pensar desligado envia think=false (so para modelos com raciocinio; nao use com gemma3 e llama3.2).

Um modelo por vez: antes de cada modelo o script descarrega o anterior (memoria limpa) e a 1a chamada de cada
modelo e 'a frio' (inclui a carga). O `bge-m3` fica carregado pela propria busca (cenario real do app).

Saidas em resultados/ (fora do Git): medicao_llm_<rotulo>.csv, medicao_llm_<rotulo>.md (resumo contra as
metas da decisao 5) e medicao_llm_<rotulo>_textos.md (todas as respostas, para a leitura humana de qualidade).
O resumo NAO mede o desenho da tela do Streamlit: so Buscador + llm.
"""
import argparse
import statistics
import sys
import time

import requests

from autopecas import llm
from autopecas.config import CATALOGO_DB, RAIZ, RESULTADOS
from autopecas.evaluation import medicao_llm as ml
from autopecas.evaluation.congelamento import commit_atual
from autopecas.retrieval.ollama_embed import _host_padrao
from autopecas.search import Buscador
from autopecas.store import Catalogo

# Consultas do tipo balcao: exemplos do app, perguntas do gabarito do Basico (texto das perguntas, sem
# dados) e uma peca que o catalogo nao tem (o LLM deve dizer que nao encontrou).
CONSULTAS = [
    "cabo capo uno", "palheta s435", "filtro oleo gol", "amortecedor dianteiro fusca",
    "retrovisor do uno lado esquerdo", "bandeja dianteira do pampa", "luz da placa do gol",
    "jogo de juntas do motor do gol", "palheta de 24 polegadas", "pastilha de freio opala",
    "cabo de freio", "ignição eletrônica",
]


def _versao_ollama(host: str) -> str:
    try:
        return str(requests.get(f"{host}/api/version", timeout=2).json().get("version", "n/d"))
    except Exception:
        return "n/d"


def _carregados(host: str) -> list[str]:
    try:
        return [m.get("name", "?") for m in requests.get(f"{host}/api/ps", timeout=2).json().get("models", [])]
    except Exception:
        return []


def _mediana_ms(fn, repeticoes: int) -> float:
    fn()  # aquecimento: o vetor da consulta entra no cache
    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        fn()
        tempos.append((time.perf_counter() - t0) * 1000)
    return statistics.median(tempos)


def _linha(modelo, consulta, a_frio, res, exp) -> dict:
    est = exp.estatisticas if exp is not None else None
    return {
        "modelo": modelo, "consulta": consulta, "a_frio": a_frio, "modo": res.modo, "n_itens": len(res),
        "lista_ms": res.latencia_ms,
        "primeira_palavra_s": exp.t_primeira_palavra if exp is not None else None,
        "total_s": exp.t_total if exp is not None else None,
        "carga_s": est.carga_s if est else None,
        "prompt_tokens": est.prompt_tokens if est else None,
        "gerados": est.gerados if est else None,
        "eval_rate": est.eval_rate if est else None,
        "prompt_rate": est.prompt_rate if est else None,
        "concluida": bool(exp and exp.concluida),
        "truncada": bool(exp and exp.truncada),
        "bloqueada": bool(exp and exp.bloqueada),
        "violacoes": "; ".join(f"{v.tipo}: {v.trecho}" for v in exp.violacoes) if exp is not None else "",
        "erro": exp.erro if exp is not None else "sem itens na lista",
        "texto": exp.texto if exp is not None else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", default=["gemma3:1b", "llama3.2:3b"])
    ap.add_argument("--consultas-arquivo", help="uma consulta por linha (substitui a lista padrao)")
    ap.add_argument("--k", type=int, default=llm.MAX_ITENS, help="itens da lista (o prompt usa ate MAX_ITENS)")
    ap.add_argument("--repeticoes", type=int, default=20, help="repeticoes para o custo da regra de veiculo")
    ap.add_argument("--pensar", choices=["padrao", "desligado"], default="padrao",
                    help="desligado = envia think=false (modelos com a marca thinking)")
    ap.add_argument("--prompt", choices=list(llm.VARIANTES), default="a",
                    help="variante do prompt: a (so instrucao), b (instrucao curta + 2 exemplos) ou c (3 exemplos, 3 itens)")
    ap.add_argument("--rotulo", default="etapa5", help="sufixo dos arquivos de saida")
    ap.add_argument("--db", default=str(CATALOGO_DB), help="catalogo SQLite")
    ap.add_argument("--sem-llm", action="store_true", help="mede so a lista e o custo da regra de veiculo")
    args = ap.parse_args()

    consultas = CONSULTAS
    if args.consultas_arquivo:
        with open(args.consultas_arquivo, encoding="utf-8") as f:
            consultas = [ln.strip() for ln in f if ln.strip()]
    if not consultas:
        sys.exit("Nenhuma consulta.")
    try:
        with Catalogo(args.db) as cat:
            itens = cat.itens()
    except Exception as e:  # banco ausente ou ilegivel
        sys.exit(f"Não consegui abrir o catálogo em {args.db}: {e}\nRode antes: python scripts/06_importar_catalogo.py")
    if itens.empty:
        sys.exit("Catálogo vazio. Rode antes: python scripts/06_importar_catalogo.py")

    host = _host_padrao()
    buscador = Buscador.criar(itens, banco=args.db)
    buscador.verificar_servidor()  # sonda direta (GET /api/tags); com vetores em cache, so a sonda diz a verdade
    if not buscador.degradado:
        buscador.aquecer()  # carrega o bge-m3 na RAM, como o app faz ao abrir
    print(f"Catálogo: {len(itens)} itens · modo de busca: {buscador.modo} · Ollama {_versao_ollama(host)}")
    if buscador.degradado:
        print("AVISO: o Ollama não respondeu para o bge-m3; a lista está em modo lexical (as metas valem para o híbrido).")

    # 1) Lista de cada consulta (1a vez: inclui o embedding da consulta nova) e custo da regra de veiculo.
    resultados, custos = {}, []
    for c in consultas:
        resultados[c] = buscador.buscar(c, k=args.k)
        if resultados[c].consulta is not None and resultados[c].consulta.tem_veiculo:
            custos.append({
                "consulta": c,
                "ms_sem": _mediana_ms(lambda: buscador.buscar(c, k=args.k, por_veiculo=False), args.repeticoes),
                "ms_com": _mediana_ms(lambda: buscador.buscar(c, k=args.k, por_veiculo=True), args.repeticoes),
            })
    print(f"Lista: {len(consultas)} consultas · custos da regra de veículo medidos em {len(custos)}")

    # 2) LLM, um modelo por vez.
    linhas, usados, anterior = [], [], None
    if not args.sem_llm:
        for modelo in args.modelos:
            if not llm.modelo_disponivel(modelo, host=host):
                print(f"[{modelo}] não está instalado: ollama pull {modelo}  (pulando)")
                continue
            if anterior:
                llm.descarregar(anterior, host=host)
            llm.descarregar(modelo, host=host)  # garante a 1a chamada 'a frio'
            usados.append(modelo)
            for i, c in enumerate(consultas):
                res = resultados[c]
                exp = llm.explicar(c, res.itens, modelo=modelo, host=host, pensar=False if args.pensar == "desligado" else None, variante=args.prompt)
                if exp is not None:
                    for _ in exp:
                        pass
                linha = _linha(modelo, c, i == 0, res, exp)
                linhas.append(linha)
                estado = ("ERRO" if linha["erro"] else "BLOQUEADA" if linha["bloqueada"] else "ok")
                pp = linha["primeira_palavra_s"]
                print(f"[{modelo}] {c!r}: {estado} · 1ª palavra {pp if pp is None else round(pp, 1)} s"
                      f" · total {round(linha['total_s'], 1) if linha['total_s'] is not None else 'n/d'} s"
                      f"{' (a frio)' if i == 0 else ''}")
                if i == 0:
                    print(f"  carregados no Ollama agora: {', '.join(_carregados(host)) or 'n/d'}")
            anterior = modelo
        if anterior:
            llm.descarregar(anterior, host=host)

    if not args.sem_llm and not usados:
        sys.exit("Nenhum modelo foi medido (nenhum instalado). Baixe com ollama pull e rode de novo.")

    # 3) Saidas.
    base = RESULTADOS / f"medicao_llm_{args.rotulo}"
    cabecalho = {
        "rotulo": args.rotulo, "commit": commit_atual(RAIZ), "modelos": usados or args.modelos,
        "n_consultas": len(consultas), "k": args.k, "pensar": args.pensar, "prompt": args.prompt, "max_itens": llm.max_itens_da_variante(args.prompt), "versao_ollama": _versao_ollama(host),
    }
    if not linhas:  # --sem-llm: so a lista (uma linha por consulta, sem modelo)
        linhas = [_linha("(sem LLM)", c, False, resultados[c], None) for c in consultas]
    ml.escrever_csv(linhas, base.with_suffix(".csv"))
    relatorio = ml.relatorio_md(linhas, custos, cabecalho)
    base.with_suffix(".md").write_text(relatorio, encoding="utf-8")
    base.with_name(base.name + "_textos.md").write_text(ml.relatorio_textos(linhas), encoding="utf-8")
    print("\n" + relatorio)
    print(f"\nArquivos: {base.with_suffix('.csv')}, {base.with_suffix('.md')}, {base.name}_textos.md (em resultados/)")


if __name__ == "__main__":
    main()
