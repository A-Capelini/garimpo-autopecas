import sqlite3

import numpy as np

from garimpo.recuperacao.embeddings import OllamaEmbedder


class TransporteFalso:
    """Simula o servidor Ollama: devolve vetor determinístico por texto."""

    def __init__(self) -> None:
        self.chamadas = 0

    def post(self, url: str, json: dict) -> dict:
        self.chamadas += 1
        vetores = []
        for texto in json["input"]:
            rng = np.random.default_rng(sum(ord(c) for c in texto) % 1000)
            vetores.append(rng.random(8).tolist())
        return {"embeddings": vetores}


def test_embed_chama_transporte_na_primeira_vez():
    transporte = TransporteFalso()
    embedder = OllamaEmbedder(transporte=transporte)
    vetor = embedder.embed("filtro de oleo")
    assert vetor.shape == (8,)
    assert transporte.chamadas == 1


def test_embed_usa_cache_na_segunda_vez():
    transporte = TransporteFalso()
    conn = sqlite3.connect(":memory:")
    embedder = OllamaEmbedder(transporte=transporte, conn_cache=conn)
    v1 = embedder.embed("filtro de oleo")
    v2 = embedder.embed("filtro de oleo")
    assert transporte.chamadas == 1
    assert np.array_equal(v1, v2)


def test_embed_lote_mantem_ordem_e_agrupa_em_uma_chamada():
    transporte = TransporteFalso()
    embedder = OllamaEmbedder(transporte=transporte)
    vetores = embedder.embed_lote(["a", "b", "a"])
    assert vetores.shape == (3, 8)
    assert np.array_equal(vetores[0], vetores[2])
    assert transporte.chamadas == 1
