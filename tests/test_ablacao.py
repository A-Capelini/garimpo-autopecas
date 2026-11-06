import pandas as pd

from autopecas.data.quality import normalizar_texto
from autopecas.evaluation.ablacao import ablacao_lexical


def test_normalizacao_oficial_preserva_comportamento():
    assert normalizar_texto("CB.CAPO UNO/PRE 84>") == "cb capo uno/pre 84>"
    assert normalizar_texto("FILTRO OLEO -:") == "filtro oleo"


def test_normalizacao_simples_troca_simbolos_por_espaco():
    assert normalizar_texto("CB.CAPO UNO/PRE 84>", manter_simbolos=False) == "cb capo uno pre 84"
    assert normalizar_texto("FILTRO OLEO -:", manter_simbolos=False) == "filtro oleo"


def test_ablacao_gera_8_celulas_distintas_por_nivel():
    linhas = []
    for p in range(12):
        for sup in ("A", "E" if p % 2 else "F"):
            linhas.append({"fornecedor": sup, "gtin_chave": f"G{p}", "descricao": f"CABO CAPO/UNO {p}>"})
    df = pd.DataFrame(linhas)
    tab = ablacao_lexical(df, niveis=["Avançado"])
    assert len(tab) == 8
    assert not tab.duplicated(["normalizacao", "min_df", "sublinear_tf"]).any()
    assert set(tab["normalizacao"]) == {"oficial", "simples"}
