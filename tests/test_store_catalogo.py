import sqlite3

import numpy as np
import pandas as pd
import pytest

from autopecas.retrieval.ollama_embed import OllamaEmbeddingRetriever
from autopecas.store import Catalogo


def linha(forn, cod, desc, preco=10.0, gtin=None, ncm="87089990", texto=None):
    return {
        "codigo_interno": cod, "gtin": gtin, "gtin_status": "valido" if gtin else "ausente",
        "gtin_chave": gtin, "descricao": desc, "preco": preco, "ncm": ncm, "fornecedor": forn,
        "texto_busca": texto if texto is not None else desc.lower(),
    }


def df_de(*linhas):
    return pd.DataFrame(list(linhas))


@pytest.fixture
def relogio():
    """Relogio controlavel: cada chamada devolve o instante atual definido pelo teste."""
    estado = {"t": "2026-11-06T10:00:00+00:00"}

    def agora():
        return estado["t"]

    agora.estado = estado
    return agora


@pytest.fixture
def cat(relogio):
    c = Catalogo(":memory:", agora=relogio)
    yield c
    c.fechar()


def test_chave_composta_aceita_mesmo_codigo_em_fornecedores_diferentes(cat):
    r = cat.importar(df_de(linha("A", "100", "cabo capo"), linha("B", "100", "palheta")))
    assert r.inseridos == 2
    itens = cat.itens()
    assert sorted(itens["fornecedor"]) == ["A", "B"]


def test_importacao_inicial_e_colunas(cat):
    cat.importar(df_de(linha("A", "1", "cabo capo uno", 16.5, "7898110832573")))
    itens = cat.itens()
    assert list(itens.columns[:9]) == [
        "codigo_interno", "gtin", "gtin_status", "gtin_chave", "descricao", "preco", "ncm",
        "fornecedor", "texto_busca",
    ]
    assert itens.loc[0, "preco"] == 16.5 and itens.loc[0, "gtin"] == "7898110832573"
    assert bool(itens.loc[0, "ativo"]) is True


def test_reimportar_igual_nao_altera_nada(cat, relogio):
    entrada = df_de(linha("A", "1", "cabo"), linha("A", "2", "palheta"))
    cat.importar(entrada)
    relogio.estado["t"] = "2026-11-07T10:00:00+00:00"
    r = cat.importar(entrada)
    assert (r.inseridos, r.atualizados, r.inalterados, r.textos_novos) == (0, 0, 2, 0)
    assert set(cat.itens()["atualizado_em"]) == {"2026-11-06T10:00:00+00:00"}


def test_mudanca_de_preco_atualiza_sem_gerar_texto_novo(cat, relogio):
    cat.importar(df_de(linha("A", "1", "cabo", 10.0)))
    relogio.estado["t"] = "2026-11-07T10:00:00+00:00"
    r = cat.importar(df_de(linha("A", "1", "cabo", 12.0)))
    assert (r.atualizados, r.precos_alterados, r.textos_novos) == (1, 1, 0)
    itens = cat.itens()
    assert itens.loc[0, "preco"] == 12.0
    assert itens.loc[0, "criado_em"] == "2026-11-06T10:00:00+00:00"
    assert itens.loc[0, "atualizado_em"] == "2026-11-07T10:00:00+00:00"


def test_descricao_alterada_gera_texto_novo(cat):
    cat.importar(df_de(linha("A", "1", "cabo capo")))
    r = cat.importar(df_de(linha("A", "1", "cabo do capo uno")))
    assert (r.atualizados, r.precos_alterados, r.textos_novos) == (1, 0, 1)


def test_texto_ja_existente_em_outro_item_nao_conta_como_novo(cat):
    cat.importar(df_de(linha("A", "1", "cabo capo")))
    r = cat.importar(df_de(linha("A", "1", "cabo capo"), linha("B", "9", "cabo capo")))
    assert (r.inseridos, r.textos_novos) == (1, 0)


def test_item_novo_entre_existentes(cat):
    cat.importar(df_de(linha("A", "1", "cabo")))
    r = cat.importar(df_de(linha("A", "1", "cabo"), linha("A", "2", "palheta")))
    assert (r.inseridos, r.inalterados, r.textos_novos) == (1, 1, 1)
    assert cat.contar() == {"ativos": 2, "inativos": 0}


def test_sem_sincronizar_nada_e_desativado(cat):
    cat.importar(df_de(linha("A", "1", "cabo"), linha("A", "2", "palheta")))
    r = cat.importar(df_de(linha("A", "1", "cabo")))
    assert r.desativados == 0
    assert cat.contar() == {"ativos": 2, "inativos": 0}


def test_sincronizar_desativa_so_fornecedores_presentes_e_reativa(cat):
    cat.importar(df_de(linha("A", "1", "cabo"), linha("A", "2", "palheta"), linha("B", "7", "filtro")))
    r = cat.importar(df_de(linha("A", "1", "cabo")), sincronizar=True)
    assert r.desativados == 1  # A/2 some; B nao esta na planilha, nao e tocado
    assert cat.contar() == {"ativos": 2, "inativos": 1}
    assert sorted(cat.itens()["descricao"]) == ["cabo", "filtro"]
    assert len(cat.itens(apenas_ativos=False)) == 3
    r = cat.importar(df_de(linha("A", "1", "cabo"), linha("A", "2", "palheta")), sincronizar=True)
    assert (r.reativados, r.atualizados, r.desativados) == (1, 0, 0)
    assert cat.contar() == {"ativos": 3, "inativos": 0}


def test_reativar_com_dado_alterado_conta_nos_dois(cat):
    cat.importar(df_de(linha("A", "1", "cabo", 10.0), linha("A", "2", "x")))
    cat.importar(df_de(linha("A", "2", "x")), sincronizar=True)
    r = cat.importar(df_de(linha("A", "1", "cabo", 11.0), linha("A", "2", "x")))
    assert (r.reativados, r.atualizados, r.precos_alterados) == (1, 1, 1)


def test_duplicata_na_entrada_e_recusada_e_nada_e_gravado(cat):
    with pytest.raises(ValueError, match="repetem"):
        cat.importar(df_de(linha("A", "1", "cabo"), linha("A", "1", "outro")))
    assert cat.contar() == {"ativos": 0, "inativos": 0}


def test_colunas_ausentes_e_campos_obrigatorios(cat):
    with pytest.raises(ValueError, match="colunas ausentes"):
        cat.importar(pd.DataFrame({"codigo_interno": ["1"]}))
    ruim = df_de(linha("A", "1", "cabo"))
    ruim.loc[0, "descricao"] = None
    with pytest.raises(ValueError, match="descricao"):
        cat.importar(ruim)
    vazio = df_de(linha("A", " ", "cabo"))
    with pytest.raises(ValueError, match="vazio"):
        cat.importar(vazio)


def test_ausentes_viram_null_e_voltam_como_ausentes(cat):
    e = df_de(linha("A", "1", "cabo", preco=np.nan, gtin=None))
    cat.importar(e)
    itens = cat.itens()
    assert pd.isna(itens.loc[0, "preco"]) and pd.isna(itens.loc[0, "gtin"])
    # reimportar o mesmo registro (NaN == NaN no banco) nao conta como mudanca
    assert cat.importar(e).inalterados == 1


def test_ordem_de_insercao_e_estavel(cat):
    cat.importar(df_de(linha("B", "2", "b"), linha("A", "1", "a"), linha("A", "3", "c")))
    cat.importar(df_de(linha("A", "1", "a novo texto")))  # update nao muda a posicao
    assert cat.itens()["codigo_interno"].tolist() == ["2", "1", "3"]


def test_persiste_em_arquivo_e_esquema_versionado(tmp_path):
    caminho = tmp_path / "sub" / "catalogo.db"
    with Catalogo(caminho) as c:
        c.importar(df_de(linha("A", "1", "cabo")))
    with Catalogo(caminho) as c:
        assert c.contar()["ativos"] == 1
    conn = sqlite3.connect(caminho)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="mais novo"):
        Catalogo(caminho)


def test_absorver_cache_e_idempotente(tmp_path, cat):
    antigo = tmp_path / "embeddings.db"
    conn = sqlite3.connect(antigo)
    conn.execute(
        "CREATE TABLE embeddings_cache (modelo TEXT NOT NULL, texto_hash TEXT NOT NULL, "
        "vetor BLOB NOT NULL, PRIMARY KEY (modelo, texto_hash))"
    )
    conn.executemany(
        "INSERT INTO embeddings_cache VALUES (?, ?, ?)",
        [("m", f"h{i}", np.ones(4, dtype=np.float32).tobytes()) for i in range(5)],
    )
    conn.commit()
    conn.close()
    assert cat.absorver_cache(antigo) == 5
    assert cat.absorver_cache(antigo) == 0
    assert cat.contar_vetores("m") == 5 and cat.contar_vetores("outro") == 0


class _Resp:
    status_code = 200

    def __init__(self, d):
        self._d = d
        self.text = ""

    def json(self):
        return self._d


class _Sessao:
    def __init__(self):
        self.pedidos = []

    def post(self, url, json, timeout):
        self.pedidos.append(list(json["input"]))
        return _Resp({"embeddings": [[float(len(t)), 1.0, 2.0] for t in json["input"]]})


def test_vetores_vivem_no_mesmo_arquivo_do_catalogo(tmp_path):
    caminho = tmp_path / "catalogo.db"
    sessao = _Sessao()
    with Catalogo(caminho) as c:
        c.importar(df_de(linha("A", "1", "cabo capo"), linha("A", "2", "palheta")))
        textos = c.itens()["texto_busca"].tolist()
        emb = OllamaEmbeddingRetriever("m", banco=caminho, sessao=sessao).fit(textos)
        assert c.contar_vetores("m") == 2
        # item novo: so o texto novo vai ao Ollama; os demais vem do cache do mesmo arquivo
        c.importar(df_de(linha("A", "1", "cabo capo"), linha("A", "2", "palheta"), linha("A", "3", "filtro")))
        emb.fit(c.itens()["texto_busca"].tolist())
        assert sessao.pedidos[-1] == ["filtro"]
        assert c.contar_vetores("m") == 3
        emb.fechar()


def test_remover_vetores(tmp_path):
    caminho = tmp_path / "catalogo.db"
    with Catalogo(caminho) as c:
        emb = OllamaEmbeddingRetriever("m", banco=caminho, sessao=_Sessao()).fit(["a", "b"])
        h = emb.hash_documento("a")
        assert c.remover_vetores("m", [h, "inexistente"]) == 1
        assert c.contar_vetores("m") == 1
        emb.fechar()


def test_simular_devolve_o_mesmo_relatorio_sem_gravar(cat):
    cat.importar(df_de(linha("A", "1", "cabo", 10.0), linha("A", "2", "palheta")))
    nova = df_de(linha("A", "1", "cabo", 12.0), linha("A", "3", "filtro"))
    antes = cat.itens().copy()
    previa = cat.importar(nova, sincronizar=True, simular=True)
    assert cat.contar() == {"ativos": 2, "inativos": 0}
    pd.testing.assert_frame_equal(cat.itens(), antes)
    real = cat.importar(nova, sincronizar=True)
    assert previa == real
    assert (real.inseridos, real.precos_alterados, real.desativados) == (1, 1, 1)
