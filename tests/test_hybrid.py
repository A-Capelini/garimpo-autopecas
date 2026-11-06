import numpy as np
import pandas as pd
import pytest

from autopecas.evaluation.runner import avaliar_niveis
from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama


class Fixo:
    """Recuperador com matriz de scores fixa (linhas = consultas por posicao)."""

    corpus_dependente = True

    def __init__(self, M, nome="fixo"):
        self.M, self.nome = np.asarray(M, dtype=float), nome

    def fit(self, textos):
        self.n = len(list(textos))

    def score_rows(self, linhas):
        return self.M[list(linhas)].copy()

    def score_texts(self, consultas):
        return self.M[: len(consultas)].copy()


LEX = [[9.0, 0.9, 0.5, 0.1]]  # consulta = doc 0 (ele mesmo com score alto: deve ser ignorado)
SEM = [[9.0, 0.1, 0.5, 0.9]]


def hib(**kw):
    return HybridRetriever(Fixo(LEX), Fixo(SEM), **kw).fit(["a", "b", "c", "d"])


def test_rrf_formula_e_exclusao_da_propria_consulta():
    h = hib(peso_lexical=3.0, peso_semantico=1.0, k_rrf=60, profundidade=None)
    F = h.score_rows([0])
    # sem a consulta: lex -> d1:1 d2:2 d3:3 ; sem -> d3:1 d2:2 d1:3
    assert F[0, 0] == -np.inf
    assert F[0, 1] == pytest.approx(3 / 61 + 1 / 63)
    assert F[0, 2] == pytest.approx(3 / 62 + 1 / 62)
    assert F[0, 3] == pytest.approx(3 / 63 + 1 / 61)


def test_rrf_profundidade_zera_quem_esta_fora_do_topo():
    F = hib(peso_lexical=3.0, peso_semantico=1.0, k_rrf=60, profundidade=1).score_rows([0])
    assert F[0, 1] == pytest.approx(3 / 61)  # topo do lexical; fora do topo do semantico
    assert F[0, 3] == pytest.approx(1 / 61)  # topo do semantico apenas
    assert F[0, 2] == 0.0


def test_rrf_sem_peso_semantico_preserva_a_ordem_lexical():
    F = hib(peso_semantico=0.0, profundidade=None).score_rows([0])
    assert list(np.argsort(-F[0])) == [1, 2, 3, 0]


def test_soma_extremos_reproduzem_cada_fonte():
    so_lex = hib(metodo="soma", peso_lexical=1.0).score_rows([0])
    so_sem = hib(metodo="soma", peso_lexical=0.0).score_rows([0])
    assert list(np.argsort(-so_lex[0])) == [1, 2, 3, 0]
    assert list(np.argsort(-so_sem[0])) == [3, 2, 1, 0]
    assert so_lex[0, 0] == so_sem[0, 0] == -np.inf


def test_soma_minmax_ignora_a_consulta():
    F = hib(metodo="soma", peso_lexical=0.5).score_rows([0])
    # lex sem consulta: 0.9,0.5,0.1 -> 1, .5, 0 ; sem: 0.1,0.5,0.9 -> 0, .5, 1
    assert F[0, 1] == pytest.approx(0.5) and F[0, 2] == pytest.approx(0.5) and F[0, 3] == pytest.approx(0.5)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        HybridRetriever(Fixo(LEX), Fixo(SEM), metodo="outro")
    with pytest.raises(ValueError):
        HybridRetriever(Fixo(LEX), Fixo(SEM), metodo="soma", peso_lexical=3.0)


class SemOff(Fixo):
    def fit(self, textos):
        raise ErroOllama("fora do ar")


def test_fallback_lexical_so_quando_pedido():
    with pytest.raises(ErroOllama):
        HybridRetriever(Fixo(LEX), SemOff(SEM)).fit(["a", "b", "c", "d"])
    h = HybridRetriever(Fixo(LEX), SemOff(SEM), fallback_lexical=True).fit(["a", "b", "c", "d"])
    assert h.degradado
    F = h.score_rows([0])
    assert F[0, 0] == -np.inf and list(np.argsort(-F[0])) == [1, 2, 3, 0]


class SemQuedaNaConsulta(Fixo):
    def score_texts(self, consultas):
        raise ErroOllama("caiu")


def test_fallback_na_consulta_do_app():
    h = HybridRetriever(Fixo(LEX), SemQuedaNaConsulta(SEM), fallback_lexical=True).fit(["a", "b", "c", "d"])
    S = h.score_texts(["x"])
    assert h.degradado and np.allclose(S, np.asarray(LEX))
    h2 = HybridRetriever(Fixo(LEX), SemQuedaNaConsulta(SEM)).fit(["a", "b", "c", "d"])
    with pytest.raises(ErroOllama):
        h2.score_texts(["x"])


# ---------- integracao com o avaliador por nivel ----------
class SemHash:
    nome = "sem-hash"
    corpus_dependente = False

    def fit(self, textos):
        textos = list(textos)
        E = np.zeros((len(textos), 64))
        for i, t in enumerate(textos):
            for k in range(len(t) - 2):
                E[i, sum(ord(c) * (n + 1) for n, c in enumerate(t[k : k + 3])) % 64] += 1
        n = np.linalg.norm(E, axis=1, keepdims=True)
        self._E = E / np.where(n == 0, 1, n)

    def score_rows(self, linhas):
        return self._E[list(linhas)] @ self._E.T

    def score_texts(self, consultas):
        raise NotImplementedError


def catalogo():
    letras = "ABCDEFGHIJKL"
    bases = ["cabo capo uno", "filtro oleo gol", "amortecedor dianteiro fusca", "vela ignicao palio"]
    linhas = []
    for p in range(40):
        b = bases[p % 4]
        for j, sup in enumerate((letras[p % 12], letras[(5 * p + 3) % 12])):
            t = f"{b} {p}" if j == 0 else f"{b.split()[0][:2]} {' '.join(b.split()[1:])} cod{p}"
            linhas.append({"fornecedor": sup, "gtin_chave": f"G{p}", "texto_busca": t})
    return pd.DataFrame(linhas)


def test_hibrido_roda_no_avaliador_por_nivel_e_sem_peso_semantico_iguala_o_lexical():
    df = catalogo()
    fab = lambda: HybridRetriever(TfidfRetriever(min_df=1), SemHash(), peso_semantico=0.0)
    h = avaliar_niveis(df, fab)
    lex = avaliar_niveis(df, lambda: TfidfRetriever(min_df=1))
    cols = ["n_docs", "n_consultas", "hit@1", "hit@5", "hit@10"]
    pd.testing.assert_frame_equal(h[cols], lex[cols])
    assert h["retriever"].iloc[0].startswith("hibrido-rrf")
