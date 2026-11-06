import numpy as np

from garimpo.recuperacao.semantica import BuscaSemantica, fusao_rrf


def test_buscar_vetor_retorna_mais_proximo_primeiro():
    busca = BuscaSemantica()
    busca.indexar(["a", "b", "c"], np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32))
    assert busca.buscar_vetor(np.array([1, 0], dtype=np.float32), k=2) == ["a", "c"]


def test_fusao_rrf_premia_consenso():
    # "x" aparece bem nos dois rankings; "y" só no topo de um
    assert fusao_rrf([["y", "x", "z"], ["x", "z", "w"]])[0] == "x"


def test_fusao_rrf_peso_favorece_primeiro_ranking():
    assert fusao_rrf([["a", "b"], ["b", "a"]], pesos=[3.0, 1.0])[0] == "a"


def test_reranquear_ordena_candidatos_por_similaridade():
    busca = BuscaSemantica()
    busca.indexar(["a", "b", "c"], np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32))
    assert busca.reranquear(np.array([0, 1], dtype=np.float32), ["a", "c", "b"]) == ["b", "c", "a"]
