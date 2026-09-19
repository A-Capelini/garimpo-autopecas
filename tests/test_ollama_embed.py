import numpy as np
import pytest
import requests

from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever


class RespostaFalsa:
    def __init__(self, embeddings, status=200):
        self._e, self.status_code = embeddings, status

    def json(self):
        return {"embeddings": self._e}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class SessaoFalsa:
    """Simula o Ollama: o vetor de cada texto depende do seu tamanho (determinístico)."""

    def __init__(self):
        self.chamadas = []

    def post(self, url, json, timeout):
        self.chamadas.append((url, json))
        return RespostaFalsa([[float(len(t)), 1.0, 0.0] for t in json["input"]])


def test_embeddings_em_lotes_normalizados_e_cache(tmp_path):
    sessao = SessaoFalsa()
    r = OllamaEmbeddingRetriever("modelo-x", lote=2, cache_dir=tmp_path, sessao=sessao)
    textos = ["a", "bb", "ccc", "dddd", "eeeee"]
    r.fit(textos)
    assert len(sessao.chamadas) == 3  # 5 textos / lote de 2
    assert sessao.chamadas[0][0].endswith("/api/embed")
    assert np.allclose(np.linalg.norm(r._E, axis=1), 1.0)

    # segunda indexacao usa o cache: nenhuma chamada nova
    r2 = OllamaEmbeddingRetriever("modelo-x", lote=2, cache_dir=tmp_path, sessao=sessao)
    r2.fit(textos)
    assert len(sessao.chamadas) == 3
    assert np.allclose(r._E, r2._E)

    scores = r.score_rows([0, 4])
    assert scores.shape == (2, 5)
    assert np.isclose(scores[0, 0], 1.0)


def test_modelo_inexistente_da_mensagem_util(tmp_path):
    class Sessao404:
        def post(self, url, json, timeout):
            return RespostaFalsa([], status=404)

    r = OllamaEmbeddingRetriever("nao-existe", cache_dir=tmp_path, sessao=Sessao404())
    with pytest.raises(ErroOllama, match="ollama pull nao-existe"):
        r.fit(["a"])


def test_servidor_desligado_da_mensagem_util(tmp_path):
    class SessaoOff:
        def post(self, url, json, timeout):
            raise requests.ConnectionError("recusada")

    r = OllamaEmbeddingRetriever("m", cache_dir=tmp_path, sessao=SessaoOff())
    with pytest.raises(ErroOllama, match="systemctl status ollama"):
        r.fit(["a"])
