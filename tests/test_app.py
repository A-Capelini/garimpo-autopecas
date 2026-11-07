"""Fumaca do app Streamlit (streamlit.testing). Pulado se o Streamlit nao estiver instalado.

Rodam sem Ollama: o buscador abre em modo lexical, que e o plano B da apresentacao.
"""
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
