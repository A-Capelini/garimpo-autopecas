import numpy as np
import pandas as pd
import pytest

from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama
from autopecas.search import (
    Buscador, Filtros, MODO_HIBRIDA, MODO_LEXICAL, equivalentes_por_gtin,
)
from tests.test_store_catalogo import _Sessao, linha


class SemanticoFalso:
    """Semantico de brinquedo: sobreposicao de palavras. `falhar` simula Ollama fora do ar."""

    corpus_dependente = False

    def __init__(self, falhar=False):
        self.falhar = falhar
        self.textos = []
        self.fits = 0

    def fit(self, textos):
        if self.falhar:
            raise ErroOllama("fora do ar")
        self.fits += 1
        self.textos = list(textos)

    def score_rows(self, linhas):
        raise NotImplementedError

    def score_texts(self, consultas):
        if self.falhar:
            raise ErroOllama("caiu")
        out = np.zeros((len(consultas), len(self.textos)))
        for i, q in enumerate(consultas):
            palavras = set(q.split())
            for j, t in enumerate(self.textos):
                out[i, j] = len(palavras & set(t.split())) / (len(palavras) or 1)
        return out


def catalogo():
    return pd.DataFrame([
        linha("A", "1", "CABO CAPO UNO", 16, "7898110832573", ncm="87089990", texto="cabo capo uno"),
        linha("B", "2", "CB CAPO UNO PREMIO", 20, "7898110832573", ncm="87089990", texto="cb capo uno premio"),
        linha("C", "3", "FILTRO OLEO GOL", 29, "7891234567895", ncm="84212300", texto="filtro oleo gol"),
        linha("C", "4", "PALHETA LIMPADOR S435", 35, None, ncm="85122000", texto="palheta limpador s435"),
        linha("D", "5", "AMORTECEDOR DIANTEIRO FUSCA", 89, None, ncm="87088000", texto="amortecedor dianteiro fusca"),
    ])


def buscador(sem=None, **kw):
    hib = HybridRetriever(TfidfRetriever(), sem or SemanticoFalso(), k_rrf=1, profundidade=50,
                          fallback_lexical=True, **kw)
    return Buscador(catalogo(), hib)


def test_busca_hibrida_acha_o_item_esperado_em_primeiro():
    b = buscador()
    r = b.buscar("cabo capo uno")
    assert r.modo == MODO_HIBRIDA
    assert r.itens.loc[0, "codigo_interno"] == "1" and r.itens.loc[0, "posicao"] == 1
    assert list(r.itens["posicao"]) == list(range(1, len(r) + 1))
    assert r.latencia_ms >= 0


def test_consulta_e_normalizada_como_o_texto_de_busca():
    b = buscador()
    a = b.buscar("CABO  CAPÔ/UNO!")
    c = b.buscar("cabo capo uno")
    assert a.itens["codigo_interno"].tolist() == c.itens["codigo_interno"].tolist()


def test_consulta_vazia_ou_so_simbolos_nao_lista_nada():
    b = buscador()
    assert len(b.buscar("")) == 0 and len(b.buscar(" /-> ")) == 0


def test_filtro_vale_antes_do_topk():
    b = buscador()
    # sem filtro o melhor e o item A; filtrando o fornecedor D, so o amortecedor pode aparecer
    r = b.buscar("amortecedor cabo", k=5, filtros=Filtros(fornecedores=("D",)))
    assert r.itens["fornecedor"].unique().tolist() == ["D"]
    assert r.itens.loc[0, "codigo_interno"] == "5"


def test_filtros_de_ncm_preco_e_combinados():
    b = buscador()
    assert set(b.buscar("capo uno", filtros=Filtros(categorias_ncm=("8708",))).itens["codigo_interno"]) <= {"1", "2", "5"}
    r = b.buscar("capo uno", filtros=Filtros(preco_min=18, preco_max=25))
    assert r.itens["codigo_interno"].tolist() == ["2"]
    assert len(b.buscar("capo uno", filtros=Filtros(fornecedores=("Z",)))) == 0


def test_k_limita_e_score_zero_nao_aparece():
    b = buscador()
    assert len(b.buscar("capo uno", k=1)) == 1
    assert len(b.buscar("zzzzzz qqqqq")) == 0  # nenhuma evidencia em nenhuma fonte


def test_ollama_fora_na_carga_degrada_e_reconecta_depois():
    sem = SemanticoFalso(falhar=True)
    b = buscador(sem)
    assert b.degradado and b.modo == MODO_LEXICAL
    r = b.buscar("cabo capo uno")
    assert r.modo == MODO_LEXICAL and r.itens.loc[0, "codigo_interno"] == "1"
    assert b.reconectar() is False and b.degradado  # continua fora
    sem.falhar = False
    assert b.reconectar() is True and not b.degradado
    assert b.buscar("cabo capo uno").modo == MODO_HIBRIDA


def test_ollama_cai_no_meio_da_consulta_nao_quebra():
    sem = SemanticoFalso()
    b = buscador(sem)
    assert b.buscar("cabo capo uno").modo == MODO_HIBRIDA
    sem.falhar = True
    r = b.buscar("cabo capo uno")
    assert r.modo == MODO_LEXICAL and len(r) > 0 and b.degradado
    sem.falhar = False
    assert b.reconectar() and b.buscar("cabo capo uno").modo == MODO_HIBRIDA


def test_exige_fallback_lexical():
    hib = HybridRetriever(TfidfRetriever(), SemanticoFalso(), fallback_lexical=False)
    with pytest.raises(ValueError, match="fallback_lexical"):
        Buscador(catalogo(), hib)


def test_equivalentes_por_gtin_so_com_2_fornecedores_ordenados_por_preco():
    eq = equivalentes_por_gtin(catalogo().assign(gtin_chave=lambda d: d["gtin"]))
    assert list(eq) == ["7898110832573"]
    assert eq["7898110832573"]["preco"].tolist() == [16, 20]


def test_criar_com_sessao_falsa_e_vetores_no_mesmo_banco(tmp_path):
    b = Buscador.criar(catalogo(), banco=tmp_path / "catalogo.db", sessao=_Sessao())
    assert b.modo == MODO_HIBRIDA
    assert len(b.buscar("filtro oleo gol")) > 0


def test_aquecer_ok_e_com_ollama_fora():
    assert buscador().aquecer() is True
    sem = SemanticoFalso(falhar=True)
    b = buscador(sem)
    assert b.aquecer() is False and b.degradado
    sem.falhar = False
    assert b.reconectar() and b.aquecer() is True

    ok = buscador()
    ok.hibrido.semantico.falhar = True  # cai entre a carga e o aquecimento
    assert ok.aquecer() is False and ok.degradado
