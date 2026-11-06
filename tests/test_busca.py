import numpy as np

from garimpo.busca import BuscadorHibrido, equivalentes_por_gtin


class EmbedderQuebrado:
    def embed(self, texto):
        raise ConnectionError("ollama fora do ar")


class EmbedderFixo:
    def embed(self, texto):
        return np.ones(4, dtype=np.float32)


def test_busca_cai_para_lexical_quando_ollama_falha(df_pequeno):
    matriz = np.random.default_rng(0).random((len(df_pequeno), 4)).astype(np.float32)
    buscador = BuscadorHibrido(df_pequeno, matriz, EmbedderQuebrado())
    codigos, modo = buscador.buscar("cabo capo uno", k=3)
    assert modo == "lexical"
    assert "3502" in codigos


def test_busca_hibrida_quando_embedder_responde(df_pequeno):
    matriz = np.random.default_rng(0).random((len(df_pequeno), 4)).astype(np.float32)
    buscador = BuscadorHibrido(df_pequeno, matriz, EmbedderFixo())
    _, modo = buscador.buscar("cabo capo uno", k=3)
    assert modo == "híbrida"


def test_equivalentes_agrupa_gtin_em_varios_fornecedores(df_pequeno):
    eq = equivalentes_por_gtin(df_pequeno)
    assert set(eq) == {"7898110832573"}
    assert set(eq["7898110832573"]["fornecedor"]) == {"C", "I", "J"}
