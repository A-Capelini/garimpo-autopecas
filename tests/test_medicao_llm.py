"""Testes do resumo da medicao do LLM (funcoes puras; sem Ollama)."""
import csv

import pytest

from autopecas.evaluation import medicao_llm as m


def linha(modelo="m1", consulta="q", a_frio=False, lista_ms=30.0, pp=3.0, total=9.0, concluida=True,
          bloqueada=False, erro=None, rate=20.0, prompt_rate=100.0, texto="ok", violacoes=""):
    return {
        "modelo": modelo, "consulta": consulta, "a_frio": a_frio, "modo": "híbrida", "n_itens": 5,
        "lista_ms": lista_ms, "primeira_palavra_s": pp, "total_s": total, "carga_s": 0.0,
        "prompt_tokens": 276, "gerados": 100, "eval_rate": rate, "prompt_rate": prompt_rate,
        "concluida": concluida, "bloqueada": bloqueada, "violacoes": violacoes, "erro": erro, "texto": texto,
    }


def test_resumo_separa_a_frio_do_quente_e_confere_as_metas():
    linhas = [
        linha(a_frio=True, consulta="a", pp=12.0, total=30.0),
        linha(consulta="b", pp=3.0, total=9.0),
        linha(consulta="c", pp=5.0, total=14.0, rate=18.0),
    ]
    r = m.resumir(linhas)[0]
    assert r["n"] == 3 and r["n_quentes"] == 2 and r["n_concluidas"] == 2
    assert r["pp_a_frio_s"] == 12.0 and r["pp_mediana_s"] == 4.0 and r["pp_max_s"] == 5.0
    assert r["total_mediana_s"] == 11.5 and r["total_max_s"] == 14.0
    assert r["eval_rate_mediana"] == 19.0
    assert r["atende_primeira_palavra"] is True and r["atende_total"] is True


def test_meta_nao_atendida_pelo_pior_caso_e_nao_pela_mediana():
    linhas = [linha(consulta="a", pp=2.0, total=8.0), linha(consulta="b", pp=11.0, total=31.0)]
    r = m.resumir(linhas)[0]
    assert r["pp_mediana_s"] == 6.5 and r["atende_primeira_palavra"] is False and r["atende_total"] is False


def test_bloqueadas_e_erros_contam_e_nao_entram_no_tempo_total():
    linhas = [
        linha(consulta="a", total=9.0),
        linha(consulta="b", total=1.0, concluida=False, bloqueada=True, violacoes="afirmacao: e compativel"),
        linha(consulta="c", pp=None, total=0.1, concluida=False, erro="fora do ar", rate=None, prompt_rate=None),
    ]
    r = m.resumir(linhas)[0]
    assert r["n_bloqueadas"] == 1 and r["n_erros"] == 1 and r["n_concluidas"] == 1
    assert r["total_max_s"] == 9.0  # a bloqueada (interrompida) nao encurta o total


def test_resumo_por_modelo_separado():
    linhas = [linha(modelo="a", consulta="x"), linha(modelo="b", consulta="x", total=25.0)]
    r = {x["modelo"]: x for x in m.resumir(linhas)}
    assert set(r) == {"a", "b"} and r["b"]["total_max_s"] == 25.0


def test_sem_dados_vira_nd_e_nao_quebra():
    r = m.resumir([linha(a_frio=True, pp=None, total=None, concluida=False, erro="x", rate=None, prompt_rate=None)])[0]
    assert r["atende_primeira_palavra"] is None and r["atende_total"] is None
    assert "n/d" in m.relatorio_md([linha(a_frio=True, pp=None, total=None, concluida=False, erro="x")], [], {})


def test_lista_conta_cada_consulta_uma_vez():
    linhas = [linha(modelo="a", consulta="x", lista_ms=40.0), linha(modelo="b", consulta="x", lista_ms=40.0),
              linha(modelo="a", consulta="y", lista_ms=1500.0)]
    r = m.resumir_lista(linhas)
    assert r["n"] == 2 and r["max_ms"] == 1500.0 and r["atende"] is False


def test_custo_da_regra_de_veiculo():
    r = m.resumir_custo_veiculo([
        {"consulta": "a", "ms_sem": 20.0, "ms_com": 20.3},
        {"consulta": "b", "ms_sem": 25.0, "ms_com": 25.5},
        {"consulta": "c", "ms_sem": 30.0, "ms_com": 30.1},
    ])
    assert r["n"] == 3 and r["delta_mediano_ms"] == pytest.approx(0.3) and r["delta_max_ms"] == pytest.approx(0.5)
    assert m.resumir_custo_veiculo([]) == {"n": 0}


def test_relatorios_trazem_metas_ressalvas_e_textos():
    linhas = [linha(a_frio=True, consulta="cabo capo uno", texto="Veja [A:00001]."),
              linha(consulta="filtro", bloqueada=True, concluida=False, violacoes="afirmacao: e compativel", texto="x")]
    md = m.relatorio_md(linhas, [{"consulta": "a", "ms_sem": 20.0, "ms_com": 20.3}],
                        {"rotulo": "t", "commit": "abc", "modelos": ["m1"], "n_consultas": 2, "k": 5})
    assert "≤ 1 s" in md and "[ESTIMADO]" in md and "não inclui" in md.lower()
    assert "Medido no `Buscador`, não no app." in md
    texto = m.relatorio_textos(linhas)
    assert "BLOQUEADA pela guarda" in texto and "(a frio)" in texto and "> Veja [A:00001]." in texto


def test_csv_tem_todas_as_colunas(tmp_path):
    destino = tmp_path / "sub" / "x.csv"
    m.escrever_csv([linha()], destino)
    with destino.open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert list(linhas[0].keys()) == m.COLUNAS_LINHA and linhas[0]["modelo"] == "m1"


def test_conta_recusas_com_ou_sem_acento_e_mediana_de_tokens():
    linhas = [
        linha(consulta="a", texto="Não encontrei a peça pedida."),
        linha(consulta="b", texto="NAO ENCONTREI nada."),
        linha(consulta="c", texto="O [A:1] cita o Gol."),
        linha(consulta="d", texto=None),
    ]
    r = m.resumir(linhas)[0]
    assert r["n_nao_encontrou"] == 2 and r["gerados_mediana"] == 100


def test_pior_consulta_da_lista_aparece_no_resumo_e_no_relatorio():
    linhas = [linha(consulta="rapida", lista_ms=20.0), linha(consulta="lenta", lista_ms=3000.0)]
    assert m.resumir_lista(linhas)["pior_consulta"] == "lenta"
    assert "a mais lenta: lenta" in m.relatorio_md(linhas, [], {})
    assert m.resumir_lista([linha(lista_ms=None)])["pior_consulta"] is None


def test_tabela_do_relatorio_tem_cabecalho_separador_e_linhas_alinhados():
    md = m.relatorio_md([linha(texto="ok")], [], {"prompt": "b"})
    tabela = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(tabela) == 3
    assert len({ln.count("|") for ln in tabela}) == 1
    assert "prompt: b" in md


def test_conta_respostas_cortadas_no_teto_e_mantem_tabela_alinhada():
    a, b = linha(consulta="a"), linha(consulta="b")
    b["truncada"] = True
    assert m.resumir([a, b])[0]["n_truncadas"] == 1
    md = m.relatorio_md([a, b], [], {})
    tabela = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len({ln.count("|") for ln in tabela}) == 1 and "Cortadas no teto" in md
