import pytest
import requests

from autopecas.retrieval.ollama_embed import OllamaEmbeddingRetriever


class Resp:
    def __init__(self, status=200, dados=None):
        self.status_code, self._d = status, dados or {}

    def json(self):
        return self._d


class SessaoTags:
    def __init__(self, resp=None, erro=None):
        self.resp, self.erro, self.urls = resp, erro, []

    def get(self, url, timeout):
        self.urls.append(url)
        if self.erro:
            raise self.erro
        return self.resp


def emb(sessao, modelo="bge-m3"):
    return OllamaEmbeddingRetriever(modelo, host="http://127.0.0.1:11434", sessao=sessao)


def tags(*nomes):
    return Resp(200, {"models": [{"name": n} for n in nomes]})


def test_disponivel_quando_o_modelo_esta_instalado():
    s = SessaoTags(tags("bge-m3:latest"))
    assert emb(s).disponivel() is True
    assert s.urls == ["http://127.0.0.1:11434/api/tags"]
    assert emb(SessaoTags(tags("bge-m3:latest")), "bge-m3:latest").disponivel() is True


def test_indisponivel_sem_o_modelo_com_erro_http_ou_servidor_fora():
    assert emb(SessaoTags(tags("outro:latest"))).disponivel() is False
    assert emb(SessaoTags(tags())).disponivel() is False
    assert emb(SessaoTags(Resp(500))).disponivel() is False
    assert emb(SessaoTags(erro=requests.ConnectionError("recusada"))).disponivel() is False
    assert emb(SessaoTags(erro=requests.Timeout("lento"))).disponivel() is False


def test_modelo_com_tag_diferente_nao_vale():
    assert emb(SessaoTags(tags("bge-m3:latest")), "bge-m3:567m").disponivel() is True  # mesma base
    assert emb(SessaoTags(tags("all-minilm:latest")), "bge-m3").disponivel() is False
