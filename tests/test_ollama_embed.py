import numpy as np
import pytest
import requests

from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever, _host_padrao


class RespostaFalsa:
    def __init__(self, status_code=200, dados=None):
        self.status_code = status_code
        self._dados = dados or {}
        self.text = ""

    def json(self):
        return self._dados


class SessaoFalsa:
    """Vetor deterministico por texto; registra o que foi pedido."""

    def __init__(self):
        self.pedidos: list[list[str]] = []
        self.urls: list[str] = []

    def post(self, url, json, timeout):
        self.urls.append(url)
        self.pedidos.append(list(json["input"]))
        vetores = []
        for t in json["input"]:
            rng = np.random.default_rng(abs(hash_estavel(t)) % (2**32))
            vetores.append(rng.random(8).tolist())
        return RespostaFalsa(200, {"embeddings": vetores})

    @property
    def textos_pedidos(self):
        return [t for p in self.pedidos for t in p]


def hash_estavel(t):
    return sum((i + 1) * ord(c) for i, c in enumerate(t))


class Sessao404:
    def post(self, url, json, timeout):
        return RespostaFalsa(404)


class SessaoOff:
    def post(self, url, json, timeout):
        raise requests.ConnectionError("recusada")


def novo(tmp_path, sessao, **kw):
    return OllamaEmbeddingRetriever("modelo-x", cache_dir=tmp_path, sessao=sessao, **kw)


def test_lotes_e_cache_na_segunda_vez(tmp_path):
    sessao = SessaoFalsa()
    textos = ["a", "b", "c", "d", "e"]
    r = novo(tmp_path, sessao, lote=2).fit(textos)
    assert len(sessao.pedidos) == 3  # 5 textos, lote 2
    n = len(sessao.pedidos)
    r2 = novo(tmp_path, sessao, lote=2).fit(textos)
    assert len(sessao.pedidos) == n  # tudo veio do cache
    assert np.allclose(r.score_rows([0, 1]), r2.score_rows([0, 1]))


def test_trocar_um_texto_reembeda_so_ele(tmp_path):
    sessao = SessaoFalsa()
    novo(tmp_path, sessao, lote=10).fit(["a", "b", "c"])
    sessao.pedidos.clear()
    novo(tmp_path, sessao, lote=10).fit(["a", "B-novo", "c"])
    assert sessao.textos_pedidos == ["B-novo"]


def test_subconjunto_reaproveita_cache(tmp_path):
    sessao = SessaoFalsa()
    novo(tmp_path, sessao).fit(["a", "b", "c", "d"])
    sessao.pedidos.clear()
    novo(tmp_path, sessao).fit(["b", "d"])
    assert sessao.pedidos == []


def test_duplicatas_sao_pedidas_uma_vez(tmp_path):
    sessao = SessaoFalsa()
    novo(tmp_path, sessao).fit(["a", "b", "a"])
    assert sessao.textos_pedidos == ["a", "b"]


def test_cache_separado_por_modelo(tmp_path):
    sessao = SessaoFalsa()
    OllamaEmbeddingRetriever("m1", cache_dir=tmp_path, sessao=sessao).fit(["a"])
    OllamaEmbeddingRetriever("m2", cache_dir=tmp_path, sessao=sessao).fit(["a"])
    assert len(sessao.pedidos) == 2


def test_sem_cache_nao_cria_banco_nem_reaproveita(tmp_path):
    sessao = SessaoFalsa()
    novo(tmp_path, sessao, usar_cache=False).fit(["a"])
    novo(tmp_path, sessao, usar_cache=False).fit(["a"])
    assert len(sessao.pedidos) == 2
    assert not (tmp_path / "embeddings.db").exists()


def test_scores_vetores_unitarios_e_ordem(tmp_path):
    sessao = SessaoFalsa()
    r = novo(tmp_path, sessao).fit(["a", "bb", "ccc"])
    S = r.score_rows([0, 2])
    assert S.shape == (2, 3)
    assert S[0, 0] == pytest.approx(1.0, abs=1e-5)  # unitario consigo mesmo
    assert S[1, 2] == pytest.approx(1.0, abs=1e-5)
    Sq = r.score_texts(["bb"])
    assert Sq.shape == (1, 3)
    assert int(np.argmax(Sq[0])) == 1


def test_prefixos_entram_no_texto_e_no_cache(tmp_path):
    sessao = SessaoFalsa()
    novo(tmp_path, sessao, prefixo_documento="doc: ").fit(["a"])
    assert sessao.textos_pedidos == ["doc: a"]


def test_modelo_inexistente(tmp_path):
    r = OllamaEmbeddingRetriever("nao-existe", cache_dir=tmp_path, sessao=Sessao404())
    with pytest.raises(ErroOllama, match="ollama pull nao-existe"):
        r.fit(["x"])


def test_ollama_fora_do_ar(tmp_path):
    r = OllamaEmbeddingRetriever("m", cache_dir=tmp_path, sessao=SessaoOff())
    with pytest.raises(ErroOllama, match="systemctl status ollama"):
        r.fit(["x"])


def test_falha_no_meio_preserva_lotes_ja_gravados(tmp_path):
    class Quebra(SessaoFalsa):
        def post(self, url, json, timeout):
            if len(self.pedidos) == 1:
                raise requests.ConnectionError("caiu")
            return super().post(url, json, timeout)

    sessao = Quebra()
    with pytest.raises(ErroOllama):
        novo(tmp_path, sessao, lote=2).fit(["a", "b", "c", "d"])
    ok = SessaoFalsa()
    novo(tmp_path, ok, lote=2).fit(["a", "b", "c", "d"])
    assert ok.textos_pedidos == ["c", "d"]


def test_host_padrao_usa_127_e_respeita_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _host_padrao() == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert _host_padrao() == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://servidor:9999/")
    assert _host_padrao() == "http://servidor:9999"


def test_url_chamada(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    sessao = SessaoFalsa()
    novo(tmp_path, sessao).fit(["a"])
    assert sessao.urls == ["http://127.0.0.1:11434/api/embed"]
