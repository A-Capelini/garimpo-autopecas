import numpy as np

from autopecas.evaluation.gold import Gold
from autopecas.evaluation.metrics import avaliar


def test_empate_com_ruido_de_ponto_flutuante_desempata_por_posicao():
    # docs 1 e 2 empatam de verdade; o ruido de 1e-16 NAO pode inverter a ordem
    gold = Gold(np.array([0]), (np.array([2]),))
    a = np.array([[1.0, 0.5, 0.5 + 1e-16, 0.1]])
    b = np.array([[1.0, 0.5 + 1e-16, 0.5, 0.1]])
    ra, rb = avaliar(a, gold), avaliar(b, gold)
    assert ra["mrr"] == rb["mrr"] == 0.5  # doc relevante (2) fica em 2o: posicao 1 vem antes
    assert ra["hit@1"] == rb["hit@1"] == 0.0


def test_diferenca_real_ainda_ordena():
    gold = Gold(np.array([0]), (np.array([2]),))
    s = np.array([[1.0, 0.5, 0.5001, 0.1]])
    assert avaliar(s, gold)["hit@1"] == 1.0


def test_mrr_at_10_zera_quando_o_equivalente_passa_da_posicao_10():
    gold = Gold(np.array([0, 1]), (np.array([5]), np.array([2])))
    s = np.zeros((2, 14))
    # consulta 0: o relevante (doc 5) fica na posicao 11; consulta 1: relevante em 1o
    s[0, 1:12] = np.linspace(2.0, 1.0, 11)
    s[0, 5] = 0.5
    s[0, 12] = 0.4
    s[1, 2] = 3.0
    r = avaliar(s, gold)
    assert r["hit@10"] == 0.5
    assert r["mrr@10"] == 0.5  # (0 + 1) / 2
    assert abs(r["mrr"] - (1 / 11 + 1) / 2) < 1e-9
