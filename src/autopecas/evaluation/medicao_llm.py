"""Resumo e relatorios da medicao do LLM local (Etapa 5). Motor, sem Ollama nem Streamlit.

Recebe as linhas que `scripts/12_medir_llm.py` mede (uma por modelo x consulta) e produz: o CSV, o resumo por
modelo contra as metas da decisao 5 e a leitura manual de qualidade. Tudo aqui e funcao pura (testavel).

Metas da decisao 5 `[ESTIMADO]`: lista ate 1 s; primeira palavra ate 10 s; resposta completa ate 30 s
(~100 tokens). A medicao as VERIFICA; nao as confirma de antemao.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from autopecas.llm import indica_recusa

META_LISTA_MS = 1000.0
META_PRIMEIRA_PALAVRA_S = 10.0
META_TOTAL_S = 30.0

COLUNAS_LINHA = [
    "modelo", "consulta", "a_frio", "modo", "n_itens", "lista_ms", "primeira_palavra_s", "total_s",
    "carga_s", "prompt_tokens", "gerados", "eval_rate", "prompt_rate", "concluida", "bloqueada",
    "violacoes", "erro", "truncada", "texto",
]

RUBRICA = (
    "Para cada resposta, marque: (a) não afirma compatibilidade; (b) só cita itens da lista; "
    "(c) não confunde peça (ex.: trava x cabo); (d) não inventa preço, medida ou veículo; "
    "(e) diz que não encontrou quando nenhum item serve. A guarda automática cobre só (a) e (b) "
    "e só por padrões de texto: o resto depende da leitura humana."
)


def _validos(valores):
    return [v for v in valores if v is not None]


def _mediana(valores):
    v = _validos(valores)
    return statistics.median(v) if v else None


def _maximo(valores):
    v = _validos(valores)
    return max(v) if v else None


def _nao_encontrou(texto: str) -> bool:
    return indica_recusa(texto)


def resumir(linhas: list[dict]) -> list[dict]:
    """Um dicionario por modelo. 'Quente' = todas as chamadas menos a 1a (que carrega o modelo na RAM)."""
    por_modelo: dict[str, list[dict]] = {}
    for linha in linhas:
        por_modelo.setdefault(linha["modelo"], []).append(linha)
    saida = []
    for modelo, ls in por_modelo.items():
        frias = [l for l in ls if l["a_frio"]]
        quentes = [l for l in ls if not l["a_frio"]]
        concluidas = [l for l in quentes if l["concluida"]]
        pp_max = _maximo([l["primeira_palavra_s"] for l in quentes])
        total_max = _maximo([l["total_s"] for l in concluidas])
        saida.append({
            "modelo": modelo,
            "n": len(ls),
            "n_quentes": len(quentes),
            "n_concluidas": len(concluidas),
            "n_bloqueadas": sum(1 for l in ls if l["bloqueada"]),
            "n_erros": sum(1 for l in ls if l["erro"]),
            "n_truncadas": sum(1 for l in ls if l.get("truncada")),
            "n_nao_encontrou": sum(1 for l in ls if _nao_encontrou(l["texto"])),
            "gerados_mediana": _mediana([l["gerados"] for l in concluidas]),
            "pp_a_frio_s": _mediana([l["primeira_palavra_s"] for l in frias]),
            "pp_mediana_s": _mediana([l["primeira_palavra_s"] for l in quentes]),
            "pp_max_s": pp_max,
            "total_mediana_s": _mediana([l["total_s"] for l in concluidas]),
            "total_max_s": total_max,
            "eval_rate_mediana": _mediana([l["eval_rate"] for l in concluidas]),
            "prompt_rate_mediana": _mediana([l["prompt_rate"] for l in concluidas]),
            "atende_primeira_palavra": None if pp_max is None else pp_max <= META_PRIMEIRA_PALAVRA_S,
            "atende_total": None if total_max is None else total_max <= META_TOTAL_S,
        })
    return saida


def resumir_lista(linhas: list[dict]) -> dict:
    """Tempo da lista (independe do modelo: usa uma linha por consulta distinta)."""
    vistos, valores = set(), []
    for l in linhas:
        if l["consulta"] not in vistos and l["lista_ms"] is not None:
            vistos.add(l["consulta"])
            valores.append(l["lista_ms"])
    pior = max((l for l in linhas if l["lista_ms"] is not None), key=lambda l: l["lista_ms"], default=None)
    return {
        "n": len(valores),
        "mediana_ms": _mediana(valores),
        "max_ms": _maximo(valores),
        "pior_consulta": pior["consulta"] if pior else None,
        "atende": None if not valores else max(valores) <= META_LISTA_MS,
    }


def resumir_custo_veiculo(custos: list[dict]) -> dict:
    """`custos`: dicts com consulta, ms_sem e ms_com (medianas de varias repeticoes, vetores em cache)."""
    if not custos:
        return {"n": 0}
    delta = [c["ms_com"] - c["ms_sem"] for c in custos]
    return {
        "n": len(custos),
        "mediana_sem_ms": statistics.median(c["ms_sem"] for c in custos),
        "mediana_com_ms": statistics.median(c["ms_com"] for c in custos),
        "delta_mediano_ms": statistics.median(delta),
        "delta_max_ms": max(delta),
    }


def _fmt(v, casas=1, sufixo=""):
    return "n/d" if v is None else f"{v:.{casas}f}{sufixo}"


def _sim_nao(v):
    return "n/d" if v is None else ("sim" if v else "NÃO")


def escrever_csv(linhas: list[dict], caminho: Path | str) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_LINHA, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def relatorio_md(linhas: list[dict], custos: list[dict], cabecalho: dict) -> str:
    """Resumo legivel. `cabecalho`: rotulo, commit, modelos, k, n_consultas, versao_ollama."""
    resumo = resumir(linhas)
    lista = resumir_lista(linhas)
    custo = resumir_custo_veiculo(custos)
    saida = [
        f"# Medição do LLM local (Etapa 5): {cabecalho.get('rotulo', '')}",
        "",
        f"- Código medido: `{cabecalho.get('commit', 'n/d')}` · Ollama {cabecalho.get('versao_ollama', 'n/d')}",
        f"- Modelos: {', '.join(cabecalho.get('modelos', []))} · {cabecalho.get('n_consultas', 0)} consultas · "
        f"{cabecalho.get('k', '?')} itens na lista (até {cabecalho.get('max_itens', '?')} no prompt) · "
        f"raciocínio: {cabecalho.get('pensar', 'padrao')} · prompt: {cabecalho.get('prompt', 'a')}",
        "- O que isto mede: `Buscador` + `llm` no PC de referência, com o `bge-m3` carregado. NÃO inclui o "
        "tempo de desenho da tela do Streamlit. A 1ª chamada de cada modelo é 'a frio' (carrega o modelo); "
        "as demais são 'quentes' (o prefixo do prompt de sistema pode ser reaproveitado pelo Ollama, como "
        "acontece no app).",
        "- Metas da decisão 5 `[ESTIMADO]`: lista ≤ 1 s; primeira palavra ≤ 10 s; resposta completa ≤ 30 s.",
        "",
        "## Lista de resultados (independe do LLM)",
        "",
        f"Mediana {_fmt(lista['mediana_ms'], 1, ' ms')} · máximo {_fmt(lista['max_ms'], 1, ' ms')} "
        f"em {lista['n']} consultas (a mais lenta: {lista['pior_consulta']}) · meta ≤ 1000 ms: {_sim_nao(lista['atende'])}",
        "",
        "## Por modelo",
        "",
        "| Modelo | Concluídas | Bloqueadas | Erros | 'Não encontrei' | Cortadas no teto | Tokens (med) | 1ª palavra a frio | 1ª palavra quente (med / máx) | "
        "Total quente (med / máx) | eval rate (med) | prompt rate (med) | 1ª palavra ≤ 10 s | total ≤ 30 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in resumo:
        saida.append(
            f"| {r['modelo']} | {r['n_concluidas']}/{r['n_quentes']} | {r['n_bloqueadas']} | {r['n_erros']} | "
            f"{r['n_nao_encontrou']}/{r['n']} | {r['n_truncadas']}/{r['n']} | {_fmt(r['gerados_mediana'], 0)} | "
            f"{_fmt(r['pp_a_frio_s'], 1, ' s')} | {_fmt(r['pp_mediana_s'], 1)} / {_fmt(r['pp_max_s'], 1, ' s')} | "
            f"{_fmt(r['total_mediana_s'], 1)} / {_fmt(r['total_max_s'], 1, ' s')} | "
            f"{_fmt(r['eval_rate_mediana'], 1, ' tok/s')} | {_fmt(r['prompt_rate_mediana'], 0, ' tok/s')} | "
            f"{_sim_nao(r['atende_primeira_palavra'])} | {_sim_nao(r['atende_total'])} |"
        )
    saida += ["", "## Custo da regra de veículo na busca (consultas que citam veículo; motor, vetores em cache)", ""]
    if custo["n"]:
        saida.append(
            f"Mediana sem a regra {_fmt(custo['mediana_sem_ms'], 2, ' ms')} · com a regra "
            f"{_fmt(custo['mediana_com_ms'], 2, ' ms')} · acréscimo mediano {_fmt(custo['delta_mediano_ms'], 2, ' ms')} "
            f"(máximo {_fmt(custo['delta_max_ms'], 2, ' ms')}) em {custo['n']} consultas. "
            "Medido no `Buscador`, não no app."
        )
    else:
        saida.append("Não medido nesta rodada.")
    saida += ["", "## Qualidade", "", RUBRICA, "", "Leia `medicao_llm_<rotulo>_textos.md` e anote o resultado dos itens (a) a (e).", ""]
    return "\n".join(saida)


def relatorio_textos(linhas: list[dict]) -> str:
    """Todas as respostas, por modelo, com o veredito da guarda, para a leitura humana."""
    saida = ["# Respostas do LLM (leitura manual)", "", RUBRICA, ""]
    modelo_atual = None
    for l in linhas:
        if l["modelo"] != modelo_atual:
            modelo_atual = l["modelo"]
            saida += [f"## {modelo_atual}", ""]
        estado = "ERRO: " + l["erro"] if l["erro"] else ("BLOQUEADA pela guarda: " + l["violacoes"] if l["bloqueada"] else "liberada pela guarda")
        frio = " (a frio)" if l["a_frio"] else ""
        saida += [
            f"### {l['consulta']}{frio}",
            "",
            f"- Guarda: {estado} · 1ª palavra {_fmt(l['primeira_palavra_s'], 1, ' s')} · total {_fmt(l['total_s'], 1, ' s')} "
            f"· {l['gerados'] if l['gerados'] is not None else 'n/d'} tokens",
            "- Leitura humana: (a) [ ] (b) [ ] (c) [ ] (d) [ ] (e) [ ]",
            "",
            "> " + (l["texto"].strip().replace("\n", "\n> ") if l["texto"].strip() else "(sem texto)"),
            "",
        ]
    return "\n".join(saida)
