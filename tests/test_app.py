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
from tests.test_store_catalogo import linha  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
HOME = str(RAIZ / "app" / "Home.py")
PAGINA = str(RAIZ / "app" / "pages" / "1_Atualizar_catalogo.py")


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
    comum.tentar_reconectar.clear()
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
