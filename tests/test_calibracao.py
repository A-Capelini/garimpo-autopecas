import numpy as np
import pandas as pd

from autopecas.evaluation.calibracao import calibrar
from autopecas.evaluation.gold import construir_gold
from autopecas.evaluation.metrics import avaliar, metricas_de_posicoes, primeira_posicao
from autopecas.evaluation.split import mascara_calibracao


def test_split_deterministico_independente_da_ordem_e_proporcional():
    chaves = [f"789{i:010d}" for i in range(5000)]
    m = mascara_calibracao(chaves)
    assert abs(m.mean() - 0.6) < 0.03
    assert np.array_equal(m, mascara_calibracao(chaves))
    perm = np.random.default_rng(1).permutation(len(chaves))
    m_perm = mascara_calibracao([chaves[i] for i in perm])
    assert np.array_equal(m[perm], m_perm)  # o lado depende so do GTIN


def test_mesmo_gtin_fica_do_mesmo_lado():
    m = mascara_calibracao(["A", "B", "A", "A", "B"])
    assert m[0] == m[2] == m[3] and m[1] == m[4]


def test_primeira_posicao_e_metricas_batem_com_avaliar():
    df = pd.DataFrame({
        "fornecedor": list("ABABAB"), "gtin_chave": ["g1", "g1", "g2", "g2", "g3", "g3"],
        "texto_busca": ["x"] * 6,
    })
    gold = construir_gold(df)
    rng = np.random.default_rng(0)
    S = rng.random((len(gold), len(df)))
    direto = avaliar(S, gold)
    via_pos = metricas_de_posicoes(primeira_posicao(S, gold))
    for k in ("hit@1", "hit@5", "hit@10", "mrr", "mrr@10", "n_consultas"):
        assert direto[k] == via_pos[k]


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


def catalogo():
    letras, bases = "ABCDEFGHIJKL", ["cabo capo/uno", "filtro oleo gol", "amortecedor dianteiro fusca", "vela palio>"]
    linhas = []
    for p in range(60):
        b = bases[p % 4]
        for j, sup in enumerate((letras[p % 12], letras[(5 * p + 3) % 12])):
            t = f"{b} {p}".upper() if j == 0 else f"{b.split()[0][:2]} {' '.join(b.split()[1:])} cod{p}".upper()
            linhas.append({"fornecedor": sup, "gtin_chave": f"G{p}", "descricao": t})
    return pd.DataFrame(linhas)


GRADE = dict(pesos_rrf=((1, 1),), ks_rrf=(5,), pesos_soma=(0.5,))


def test_calibrar_estrutura_e_escolha_so_pela_calibracao():
    tab, res = calibrar(catalogo(), SemHash, n_boot=200, **GRADE)
    # por (normalizacao, nivel): 1 semantico + por min_df (lexical, rrf, soma) = 7 configs; x2 splits
    assert len(tab) == 2 * 2 * 7 * 2
    assert set(tab["split"]) == {"calibracao", "teste"}
    cal = tab[(tab["nivel"] == "Avançado") & (tab["split"] == "calibracao")]
    hib = cal[~cal["config"].isin(["lexical", "semantico"])]
    melhor = hib.sort_values(["mrr", "hit@1"], ascending=False, kind="stable").iloc[0]
    esc = res["hibrido_escolhido"]
    assert (esc["normalizacao"], esc["min_df"], esc["config"]) == (melhor["normalizacao"], melhor["min_df"], melhor["config"])
    assert res["lexical_escolhido"]["config"] == "lexical"
    r = res["por_nivel"]["Avançado"]
    assert r["ic95_hit@1"][0] <= r["ic95_hit@1"][1] and r["ic95_mrr"][0] <= r["ic95_mrr"][1]
    assert res["adotar_hibrido"] == (r["delta_hit@1"] >= 0 and r["delta_mrr"] >= 0)


def test_consultas_de_calibracao_e_teste_nao_se_misturam():
    tab, _ = calibrar(catalogo(), SemHash, n_boot=50, **GRADE)
    for nivel in ("Intermediário", "Avançado"):
        t = tab[(tab["nivel"] == nivel) & (tab["config"] == "lexical") & (tab["normalizacao"] == "oficial") & (tab["min_df"] == 2)]
        n_cal = int(t[t["split"] == "calibracao"]["n_consultas"].iloc[0])
        n_teste = int(t[t["split"] == "teste"]["n_consultas"].iloc[0])
        assert n_cal > 0 and n_teste > 0
