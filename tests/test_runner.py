import numpy as np
import pandas as pd
import pytest

from autopecas.evaluation.gold import construir_gold
from autopecas.evaluation.metrics import avaliar
from autopecas.evaluation.runner import avaliar_niveis
from autopecas.retrieval.lexical import TfidfRetriever

LETRAS = "ABCDEFGHIJKL"
BASES = ["cabo capo uno", "filtro oleo gol", "amortecedor dianteiro fusca", "vela ignicao palio"]


def catalogo_sintetico() -> pd.DataFrame:
    """40 produtos, cada um vendido por 2 fornecedores distintos (A..L) com grafias diferentes."""
    linhas = []
    for p in range(40):
        base = BASES[p % len(BASES)]
        for j, sup in enumerate((LETRAS[p % 12], LETRAS[(5 * p + 3) % 12])):
            texto = f"{base} {p}" if j == 0 else f"{base.split()[0][:2]} {' '.join(base.split()[1:])} cod{p}"
            linhas.append({"fornecedor": sup, "gtin_chave": f"G{p}", "texto_busca": texto})
    for i in range(10):  # itens sem GTIN valido
        linhas.append({"fornecedor": "N", "gtin_chave": np.nan, "texto_busca": f"item solto {i}"})
    return pd.DataFrame(linhas)


class TfidfEspiao(TfidfRetriever):
    tamanhos: list[int] = []

    def fit(self, textos):
        TfidfEspiao.tamanhos.append(len(list(textos)))
        super().fit(textos)


class EmbFalso:
    """Vetores deterministicos por texto (hash de trigramas): cosseno nao depende do corpus."""

    nome = "emb-falso"
    corpus_dependente = False
    ajustes: list[int] = []

    def _vec(self, textos):
        E = np.zeros((len(textos), 64))
        for i, t in enumerate(textos):
            for k in range(len(t) - 2):
                E[i, sum(ord(c) * (n + 1) for n, c in enumerate(t[k : k + 3])) % 64] += 1
        n = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.where(n == 0, 1, n)

    def fit(self, textos):
        type(self).ajustes.append(len(list(textos)))
        self._E = self._vec(list(textos))

    def score_rows(self, linhas):
        return self._E[list(linhas)] @ self._E.T


class EmbFalsoPorNivel(EmbFalso):
    corpus_dependente = True
    ajustes: list[int] = []


def test_lexical_reajusta_o_indice_em_cada_nivel():
    df = catalogo_sintetico()
    TfidfEspiao.tamanhos = []
    tab = avaliar_niveis(df, TfidfEspiao)
    esperado = [int(n) for n in tab["n_docs"]]
    assert TfidfEspiao.tamanhos == esperado  # um ajuste por nivel, so com os docs do nivel
    assert esperado[0] < esperado[1] < esperado[2] == len(df)


def test_embeddings_indexam_o_catalogo_uma_unica_vez():
    df = catalogo_sintetico()
    EmbFalso.ajustes = []
    avaliar_niveis(df, EmbFalso)
    assert EmbFalso.ajustes == [len(df)]


def test_fatiar_equivale_a_reajustar_para_cosseno():
    df = catalogo_sintetico()
    EmbFalso.ajustes, EmbFalsoPorNivel.ajustes = [], []
    fatiado = avaliar_niveis(df, EmbFalso)
    reajustado = avaliar_niveis(df, EmbFalsoPorNivel)
    cols = ["n_docs", "n_consultas", "hit@1", "hit@5", "hit@10", "mrr"]
    pd.testing.assert_frame_equal(fatiado[cols], reajustado[cols])


def test_avancado_igual_a_avaliacao_direta_no_catalogo_inteiro():
    df = catalogo_sintetico()
    tab = avaliar_niveis(df, TfidfRetriever)
    linha = tab[tab["nivel"] == "Avançado"].iloc[0]
    r = TfidfRetriever()
    r.fit(df["texto_busca"].tolist())
    gold = construir_gold(df)
    direto = avaliar(r.score_rows(gold.consultas), gold)
    assert linha["n_consultas"] == direto["n_consultas"]
    assert linha["hit@1"] == pytest.approx(direto["hit@1"])
    assert linha["mrr"] == pytest.approx(direto["mrr"])


def test_nivel_sem_consultas_nao_quebra():
    df = pd.DataFrame(
        [
            {"fornecedor": "A", "gtin_chave": "G1", "texto_busca": "cabo capo uno"},
            {"fornecedor": "B", "gtin_chave": "G2", "texto_busca": "filtro oleo gol"},
            {"fornecedor": "C", "gtin_chave": "G3", "texto_busca": "vela palio"},
            {"fornecedor": "E", "gtin_chave": "G1", "texto_busca": "cb capo uno"},
            {"fornecedor": "F", "gtin_chave": "G2", "texto_busca": "filtro de oleo gol"},
        ]
    )
    for fab in (lambda: TfidfRetriever(min_df=1), EmbFalso):
        tab = avaliar_niveis(df, fab, niveis=["Básico", "Intermediário"])
        assert tab.loc[0, "n_consultas"] == 0 and np.isnan(tab.loc[0, "mrr"])
        assert tab.loc[1, "n_consultas"] == 4


def test_colunas_compativeis_com_avaliacao_csv():
    tab = avaliar_niveis(catalogo_sintetico(), TfidfRetriever)
    assert list(tab.columns) == [
        "retriever", "nivel", "n_docs", "n_consultas", "hit@1", "hit@5", "hit@10",
        "mrr", "seg_indexacao", "ms_por_consulta",
    ]
