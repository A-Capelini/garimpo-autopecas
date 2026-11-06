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
