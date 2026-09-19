import numpy as np
import pandas as pd

from autopecas.data.levels import fornecedores_do_nivel, mascara_nivel
from autopecas.evaluation.gold import construir_gold
from autopecas.evaluation.metrics import avaliar


def _df():
    # posicoes: 0,1,2 = mesmo GTIN em A, B, B ; 3 = GTIN so em A ; 4 = sem GTIN valido
    return pd.DataFrame(
        {
            "gtin_chave": ["111", "111", "111", "222", None],
            "fornecedor": ["A", "B", "B", "A", "C"],
        }
    )


def test_gold_usa_apenas_gtin_em_mais_de_um_fornecedor():
    g = construir_gold(_df())
    assert sorted(g.consultas.tolist()) == [0, 1, 2]
    rel = dict(zip(g.consultas.tolist(), [r.tolist() for r in g.relevantes]))
    assert rel[0] == [1, 2]  # A -> as duas linhas de B
    assert rel[1] == [0]  # B -> so a linha de A (a outra e do mesmo fornecedor)
    assert rel[2] == [0]


def test_gold_respeita_mascara():
    mascara = np.array([True, True, False, True, True])  # remove a linha 2
    g = construir_gold(_df(), mascara)
    rel = dict(zip(g.consultas.tolist(), [r.tolist() for r in g.relevantes]))
    assert rel[0] == [1]
    assert 2 not in g.consultas.tolist()


def test_metricas_ranking_perfeito_e_pior_caso():
    g = construir_gold(_df())
    n = 5
    # consulta 0 -> relevante 1 em 1o; consulta 1 -> relevante 0 em 2o; consulta 2 -> relevante 0 em 3o
    scores = np.zeros((3, n))
    scores[0, 1] = 0.9
    scores[0, 2] = 0.8
    scores[1, 3] = 0.9
    scores[1, 0] = 0.8
    scores[2, 3] = 0.9
    scores[2, 4] = 0.8
    scores[2, 0] = 0.7
    ordem = {int(c): i for i, c in enumerate(g.consultas)}
    m = avaliar(scores[[ordem[0], ordem[1], ordem[2]]], g)
    assert m["n_consultas"] == 3
    assert m["hit@1"] == 1 / 3
    assert m["hit@5"] == 1.0
    assert abs(m["mrr"] - (1 + 1 / 2 + 1 / 3) / 3) < 1e-9
    # recall@1 da consulta 0 = 1 de 2 relevantes = 0.5 ; demais = 0 e 0
    assert abs(m["recall@1"] - (0.5 + 0 + 0) / 3) < 1e-9


def test_metricas_excluem_a_propria_consulta():
    """Se a propria linha nao fosse excluida, ela ocuparia o 1o lugar e o equivalente cairia para o 2o."""
    g = construir_gold(_df())
    scores = np.full((len(g), 5), 0.1)
    scores[np.arange(len(g)), g.consultas] = 1.0  # a propria linha tem o maior score
    for i, rel in enumerate(g.relevantes):
        scores[i, rel] = 0.9  # equivalentes logo em seguida
    m = avaliar(scores, g)
    assert m["hit@1"] == 1.0


def test_niveis_sao_acumulativos():
    df = pd.DataFrame({"fornecedor": list("ABCDEFGHIJKLN") * 2})
    assert fornecedores_do_nivel(df, "Básico") == list("ABCD")
    assert fornecedores_do_nivel(df, "Intermediário") == list("ABCDEFGHI")
    assert "M" not in fornecedores_do_nivel(df, "Avançado")
    assert mascara_nivel(df, "Avançado").all()
