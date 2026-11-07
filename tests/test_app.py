"""Fumaca do app Streamlit (streamlit.testing). Pulado se o Streamlit nao estiver instalado.

Rodam sem Ollama: o buscador abre em modo lexical, que e o plano B da apresentacao.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from autopecas.store import Catalogo  # noqa: E402
from helpers import _Sessao, linha  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
HOME = str(RAIZ / "app" / "Home.py")
PAGINA = str(RAIZ / "app" / "pages" / "1_Atualizar_catalogo.py")
ADAPTACOES = str(RAIZ / "app" / "pages" / "2_Adaptacoes.py")


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Aponta o app para um catalogo temporario e um Ollama inexistente."""
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # porta fechada: conexao recusada
    monkeypatch.syspath_prepend(str(RAIZ / "app"))
    import autopecas.config as cfg
    import autopecas.search as search

    caminho = tmp_path / "catalogo.db"
    for modulo in (cfg,):
        monkeypatch.setattr(modulo, "CATALOGO_DB", caminho)
    sys.modules.pop("comum", None)
    import comum  # noqa: E402

    monkeypatch.setattr(comum, "CATALOGO_DB", caminho)
    monkeypatch.setattr(search, "CATALOGO_DB", caminho)
    comum.carregar.clear()
    comum.atualizar_status.clear()
    yield caminho, comum
    comum.carregar.clear()
    sys.modules.pop("comum", None)


def popular(caminho):
    itens = pd.DataFrame([
        linha("A", "1", "CABO CAPO UNO", 16, "7898110832573", ncm="87089990", texto="cabo capo uno"),
        linha("B", "2", "CB CAPO UNO PREMIO", 20, "7898110832573", ncm="87089990", texto="cb capo uno premio"),
        linha("C", "3", "FILTRO OLEO GOL", 29, "7891234567895", ncm="84212300", texto="filtro oleo gol"),
    ])
    with Catalogo(caminho) as c:
        c.importar(itens)


def test_home_sem_catalogo_mostra_instrucao(app_env):
    at = AppTest.from_file(HOME, default_timeout=30).run()
    assert not at.exception
    assert any("catálogo ainda está vazio" in w.value for w in at.warning)


def test_home_busca_lexical_sem_ollama(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    assert not at.exception
    assert any("offline" in m.value for m in at.markdown)  # status do Ollama
    at.text_input(key="consulta").set_value("cabo capo uno").run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Modo de busca"] == "lexical"
    assert int(metricas["Resultados"]) >= 1
    tabela = at.dataframe[0].value
    assert tabela.iloc[0]["Código"] == "1"
    # o GTIN 7898110832573 esta em A e B: aparece o bloco de equivalentes
    assert any("Mesmo produto em outros fornecedores" in s.value for s in at.subheader)


def test_home_consulta_sem_resultado(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.text_input(key="consulta").set_value("zzzzzz qqqqq").run()
    assert not at.exception
    assert any("Nenhuma peça encontrada" in w.value for w in at.warning)


def test_home_filtro_de_fornecedor(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.multiselect[0].set_value(["B"])
    at.text_input(key="consulta").set_value("capo uno").run()
    assert not at.exception
    assert at.dataframe[0].value["Fornecedor"].unique().tolist() == ["B"]


def test_pagina_de_atualizacao_abre_sem_arquivo(app_env):
    at = AppTest.from_file(PAGINA, default_timeout=30).run()
    assert not at.exception
    assert any("Escolha um arquivo" in i.value for i in at.info)


def planilha_bytes(linhas):
    """Planilha no formato real: sem cabecalho, 6 colunas (codigo, GTIN, descricao, preco, NCM, fornecedor)."""
    import io

    buf = io.BytesIO()
    pd.DataFrame(linhas).to_excel(buf, header=False, index=False)
    buf.seek(0)
    return buf


def test_pagina_de_atualizacao_previa_e_aplicacao(app_env, monkeypatch):
    import streamlit as st

    caminho, _ = app_env
    popular(caminho)
    nova = planilha_bytes([
        ["1", "7898110832573", "CABO CAPO UNO", 18, "87089990", "A"],      # preco mudou 16 -> 18
        ["4", None, "PALHETA LIMPADOR S435", 35, "85122000", "C"],          # item novo
    ])
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: nova)
    at = AppTest.from_file(PAGINA, default_timeout=60).run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert (m["Novos"], m["Atualizados"]) == ("1", "1")
    with Catalogo(caminho) as c:  # a previa nao gravou nada
        assert c.contar() == {"ativos": 3, "inativos": 0}

    at.button[0].click().run()
    assert not at.exception
    assert any("Catálogo atualizado" in s.value for s in at.success)
    with Catalogo(caminho) as c:
        itens = c.itens()
    assert len(itens) == 4
    assert float(itens.loc[itens["codigo_interno"] == "1", "preco"].iloc[0]) == 18.0


def test_pagina_de_atualizacao_recusa_planilha_invalida(app_env, monkeypatch):
    import streamlit as st

    caminho, _ = app_env
    popular(caminho)
    repetida = planilha_bytes([
        ["9", None, "X", 1, "87089990", "A"],
        ["9", None, "Y", 2, "87089990", "A"],
    ])
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: repetida)
    at = AppTest.from_file(PAGINA, default_timeout=60).run()
    assert any("repetem" in e.value for e in at.error)


def test_status_offline_mesmo_com_todos_os_vetores_em_cache(app_env):
    """Regressao (capturas de 06/11): com o cache completo, fit(), o aquecimento e consultas repetidas
    funcionam sem o Ollama; o app mostrava 'online/hibrida' com o servidor parado."""
    from autopecas.retrieval.ollama_embed import OllamaEmbeddingRetriever

    caminho, comum = app_env
    popular(caminho)
    with Catalogo(caminho) as c:
        textos = c.itens()["texto_busca"].tolist()
    pre = OllamaEmbeddingRetriever("bge-m3", banco=caminho, sessao=_Sessao())
    pre.fit(textos + ["aquecimento", "cabo capo uno"])  # tudo o que o app pediria ao Ollama
    pre.fechar()

    at = AppTest.from_file(HOME, default_timeout=60).run()
    assert not at.exception
    assert any("offline" in m.value for m in at.markdown)
    assert not any("online" in m.value and "offline" not in m.value for m in at.markdown)
    at.text_input(key="consulta").set_value("cabo capo uno").run()
    assert {m.label: m.value for m in at.metric}["Modo de busca"] == "lexical"


def test_home_mostra_o_veiculo_lido_e_a_coluna_veiculo(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.text_input(key="consulta").set_value("filtro oleo gol").run()
    assert not at.exception
    assert any("Veículo lido da consulta: gol" in c.value for c in at.caption)
    assert "Veículo" in at.dataframe[0].value.columns
    at.text_input(key="consulta").set_value("filtro oleo").run()  # sem veiculo: sem coluna
    assert "Veículo" not in at.dataframe[0].value.columns


def test_pagina_adaptacoes_sem_catalogo_e_sem_veiculo(app_env):
    at = AppTest.from_file(ADAPTACOES, default_timeout=30).run()
    assert not at.exception
    assert any("catálogo ainda está vazio" in w.value for w in at.warning)


def test_pagina_adaptacoes_exige_peca_e_veiculo_e_mostra_aviso(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(ADAPTACOES, default_timeout=60).run()
    assert not at.exception
    assert any("Confirme a aplicação" in i.value for i in at.info)   # aviso fixo
    assert any("Informe a peça e escolha o veículo" in i.value for i in at.info)


def test_pagina_adaptacoes_lista_blocos_rotulados(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(ADAPTACOES, default_timeout=60).run()
    at.text_input[0].set_value("cabo capo")
    at.selectbox[0].set_value("uno").run()
    assert not at.exception
    titulos = [s.value for s in at.subheader]
    assert any(t.startswith("Citado na descrição do fornecedor") for t in titulos)
    tabela = at.dataframe[0].value
    assert set(tabela["Situação"]) <= {"citado na descrição do fornecedor", "sugestão a validar"}
    assert "Motivo" in tabela.columns


def test_pagina_adaptacoes_recusa_ano_invalido(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(ADAPTACOES, default_timeout=60).run()
    at.text_input[0].set_value("cabo capo")
    at.selectbox[0].set_value("uno")
    at.text_input[1].set_value("88").run()
    assert any("Ano inválido" in e.value for e in at.error)


# ---------------------------------------------------------------- resumo dos primeiros itens (Etapa 5)


def _expander_llm(at):
    return next(e for e in at.expander if "IA local" in e.label)


def test_home_mostra_resumo_por_regras_sem_ollama(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.text_input(key="consulta").set_value("cabo capo uno").run()
    assert not at.exception
    assert any(s.value == "Resumo dos primeiros itens" for s in at.subheader)
    corpo = " ".join(m.value for m in at.markdown)
    assert "**A · 1**" in corpo and "CABO CAPO UNO" in corpo
    assert "R\\$ 16,00" in corpo  # cifrao escapado: nao vira LaTeX
    assert any("sem IA" in c.value for c in at.caption)
    assert any("Experimental" in c.value for c in at.caption)


def test_resumo_nao_aparece_quando_nao_ha_resultado(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.text_input(key="consulta").set_value("zzzzzz qqqqq").run()
    assert not at.exception
    assert not any(s.value == "Resumo dos primeiros itens" for s in at.subheader)


def test_ligar_ia_com_ollama_parado_avisa_e_mantem_lista_e_resumo(app_env):
    caminho, _ = app_env
    popular(caminho)
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.text_input(key="consulta").set_value("cabo capo uno").run()
    at.toggle(key="llm_ativo").set_value(True).run()
    assert not at.exception
    assert any("Ollama está parado" in w.value for w in at.warning)
    assert not any(b.key == "btn_llm" for b in at.button)  # sem modelo/servidor, nao oferece o botao
    assert len(at.dataframe) >= 1 and any("**A · 1**" in m.value for m in at.markdown)


class _RespFalsa:
    status_code = 200

    def __init__(self, textos, final=None):
        self.linhas = [json.dumps({"message": {"role": "assistant", "content": t}, "done": False}) for t in textos]
        self.linhas.append(json.dumps(final or {
            "message": {"content": ""}, "done": True, "total_duration": 4_000_000_000,
            "prompt_eval_count": 200, "prompt_eval_duration": 1_000_000_000,
            "eval_count": 40, "eval_duration": 2_000_000_000,
        }))

    def iter_lines(self):
        yield from self.linhas

    def close(self):
        pass


class _SessaoLLM:
    def __init__(self, resp):
        self.resp = resp

    def post(self, url, json=None, timeout=None, stream=False):
        return self.resp


def _ligar_llm(monkeypatch, comum, textos=None, resp=None):
    """Liga o toggle sem Ollama real: modelo 'instalado', aquecimento ok, geracao pela sessao falsa."""
    llm = comum.llm
    monkeypatch.setattr(llm, "modelo_disponivel", lambda *a, **k: True)
    monkeypatch.setattr(llm, "aquecer_modelo", lambda *a, **k: True)
    resposta = resp or _RespFalsa(textos)
    original = llm.explicar
    monkeypatch.setattr(llm, "explicar", lambda c, i, **kw: original(c, i, sessao=_SessaoLLM(resposta), host="http://x:1", **kw))


def _abrir_e_gerar(caminho, consulta="cabo capo uno"):
    at = AppTest.from_file(HOME, default_timeout=60).run()
    at.text_input(key="consulta").set_value(consulta).run()
    at.toggle(key="llm_ativo").set_value(True).run()
    at.button(key="btn_llm").click().run()
    return at


def test_resumo_por_ia_liberado_mostra_texto_aviso_e_tempos(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, ["O [A:1] por R$ 16,00 cita o Uno.", " Confira o ano."])
    at = _abrir_e_gerar(caminho)
    assert not at.exception
    corpo = " ".join(m.value for m in at.markdown)
    assert "O \\[A:1\\] por R\\$ 16,00 cita o Uno. Confira o ano." in corpo
    legendas = " ".join(c.value for c in at.caption)
    assert "Resumo gerado por IA local" in legendas and "tok/s" in legendas


def test_resumo_por_ia_com_afirmacao_e_bloqueado_e_o_texto_nao_aparece(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, ["O item é compatível com o veículo."])
    at = _abrir_e_gerar(caminho)
    assert not at.exception
    assert any("bloqueou este resumo" in w.value for w in at.warning)
    assert "compatível com o veículo" not in " ".join(m.value for m in at.markdown)


def test_resumo_por_ia_com_codigo_inventado_e_bloqueado(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, ["O [Z:99999] por R$ 5,00 cita o Uno."])
    at = _abrir_e_gerar(caminho)
    assert any("bloqueou este resumo" in w.value for w in at.warning)


def test_resumo_por_ia_cortado_no_teto_fica_na_ultima_frase(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    cortado = {"message": {"content": ""}, "done": True, "done_reason": "length", "eval_count": 120, "eval_duration": 6_000_000_000}
    _ligar_llm(monkeypatch, comum, resp=_RespFalsa(["O [A:1] cita o Uno, por R$ 16,00. O [B:2] é gen"], final=cortado))
    at = _abrir_e_gerar(caminho)
    corpo = " ".join(m.value for m in at.markdown)
    assert "por R\\$ 16,00." in corpo and "é gen" not in corpo


def test_resumo_por_ia_falha_do_ollama_nao_derruba_a_tela(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, resp=_RespFalsa([], final={"error": "x"}))
    at = _abrir_e_gerar(caminho)
    assert not at.exception
    assert any("continuam disponíveis" in w.value for w in at.warning)
    assert len(at.dataframe) >= 1


def test_resumo_por_ia_e_mantido_ao_mexer_em_outro_controle(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, ["O [A:1] cita o Uno."])
    at = _abrir_e_gerar(caminho)
    assert any("cita o Uno" in m.value for m in at.markdown)
    at.run()  # nova execucao sem clicar: o texto guardado continua para a mesma consulta e filtros
    assert any("cita o Uno" in m.value for m in at.markdown)
    at.text_input(key="consulta").set_value("filtro oleo gol").run()  # outra consulta: o resumo antigo some
    assert not any("cita o Uno" in m.value for m in at.markdown)


def test_recusa_da_ia_mostra_nota_de_que_pode_estar_errada(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, ["Não encontrei o cabo de capô entre os itens da lista."])
    at = _abrir_e_gerar(caminho)
    assert not at.exception
    assert any("pode estar errado" in i.value for i in at.info)
    assert any("Não encontrei o cabo" in m.value for m in at.markdown)  # o texto e a nota aparecem juntos


def test_resposta_normal_da_ia_nao_mostra_a_nota_de_recusa(app_env, monkeypatch):
    caminho, comum = app_env
    popular(caminho)
    _ligar_llm(monkeypatch, comum, ["O [A:1] cita o Uno, por R$ 16,00."])
    at = _abrir_e_gerar(caminho)
    assert not any("pode estar errado" in i.value for i in at.info)
