"""Extracao de veiculo e ano (src/autopecas/veiculos.py). Descricoes sinteticas (decisao 27)."""
import pandas as pd
import pytest

from autopecas.veiculos import (
    VEICULOS, aplicar_extracao, analisar_consulta, extrair, extrair_anos, extrair_veiculos, vizinhos_no_catalogo,
)


@pytest.mark.parametrize("desc, esperado", [
    ("CABO DO CAPO UNO", ("uno",)),
    ("AMORTECEDOR DIANT. GOL/VOY/PAR/SAV", ("gol", "voyage", "parati", "saveiro")),
    ("PAR DE PASTILHAS", ()),                      # "par" solto = par de pecas
    ("JOGO C/2 PAR", ()),
    ("COXIM MOTOR (PAR) GOL", ("gol",)),
    ("PARACHOQUE FUSCA", ("fusca",)),               # nao casa dentro de outra palavra
    ("TAMPA CORSA SEDAN", ("corsa",)),              # sedan so vira Fusca sem outro veiculo
    ("PARALAMA SEDAN", ("fusca",)),
    ("VELA DEL REY / CORCEL", ("del rey", "corcel")),
    ("FILTRO OLEO", ()),
])
def test_extrair_veiculos(desc, esperado):
    assert set(extrair_veiculos(desc)) == set(esperado)


@pytest.mark.parametrize("desc, ini, fim", [
    ("CABO GOL 85/90", 1985, 1990),
    ("CABO GOL 1985/1990", 1985, 1990),
    ("CORCEL DEL REY 78 A 82", 1978, 1982),
    ("FUSCA 74/", 1974, None),                      # faixa aberta no fim
    ("GOL 92>", 1992, None),
    ("UNO ATE 83", None, 1983),
    ("UNO /81", None, 1981),
    ("UNO 1988", 1988, 1988),
    ("MOTOR 1200 FUSCA", None, None),               # cilindrada nao e ano
    ("MOTOR 1300 FUSCA", None, None),
    ("PNEU 165/70 R13", None, None),                # medida de pneu
    ("PARAFUSO 12X1,5X37", None, None),
    ("FAROL 08/99 A 08/03", None, None),            # mes/ano: falha segura, nao inventa
    ("CABO", None, None),
])
def test_extrair_anos(desc, ini, fim):
    assert extrair_anos(desc) == (ini, fim)


def test_pivo_de_seculo():
    assert extrair_anos("GOL 98/02") == (1998, 2002)    # 98 -> 1998; 02 -> 2002
    assert extrair_anos("GOL 85/05") == (1985, 2005)
    assert extrair_anos("FUSCA 50/60") == (1950, 1960)  # limite inferior plausivel
    assert extrair_anos("GOL 31/35") == (None, None)    # 31 -> 1931, fora de 1950..2035: descarta


def test_extrair_combina_veiculo_e_ano():
    e = extrair("CABO CAPO UNO 85/90")
    assert e.veiculos == ("uno",) and (e.ano_ini, e.ano_fim) == (1985, 1990)
    assert e.cobre(1988) and not e.cobre(1984) and not e.cobre(1991)


def test_cobre_faixa_aberta_e_sem_ano():
    assert extrair("FUSCA 74/").cobre(2000) and not extrair("FUSCA 74/").cobre(1970)
    assert extrair("UNO ATE 83").cobre(1960) and not extrair("UNO ATE 83").cobre(1990)
    assert not extrair("UNO").cobre(1988)  # sem ano informado: nao afirma nada


def test_aplicar_extracao_cria_campos_sem_mexer_no_original():
    df = pd.DataFrame({"descricao": ["CABO UNO 85/90", "FILTRO OLEO"]})
    out = aplicar_extracao(df)
    assert list(df.columns) == ["descricao"]
    assert out["veiculos"].tolist() == [("uno",), ()]
    assert str(out["ano_ini"].dtype) == "Int64" and str(out["ano_fim"].dtype) == "Int64"
    assert out["ano_ini"].tolist()[0] == 1985 and pd.isna(out["ano_ini"].iloc[1])


def test_vocabulario_sem_siglas_ambiguas():
    for ambigua in ("par", "pst", "bras", "fire", "tipo"):
        assert ambigua not in VEICULOS and all(ambigua not in a for a in VEICULOS.values())


@pytest.mark.parametrize("consulta, peca, veiculos, ano", [
    ("cabo capo uno", "cabo capo", ("uno",), None),
    ("cabo do capo do uno 88", "cabo do capo do", ("uno",), 1988),
    ("filtro oleo gol 1995", "filtro oleo", ("gol",), 1995),
    ("palheta de 24 polegadas", "palheta de 24 polegadas", (), None),   # sem veiculo: numero e medida
    ("pastilha freio gol 24 mm", "pastilha freio 24 mm", ("gol",), None),  # numero seguido de unidade nao e ano
    ("filtro oleo", "filtro oleo", (), None),
])
def test_analisar_consulta(consulta, peca, veiculos, ano):
    c = analisar_consulta(consulta)
    assert (c.peca, c.veiculos, c.ano) == (peca, veiculos, ano)
    assert c.tem_veiculo == bool(veiculos)


def test_vizinhos_exige_pares_e_razao_minimos():
    forte = [("gol", "parati")] * 5 + [("gol",)] * 5          # 5 pares; razao 5/5 pelo menos citado
    fraco = [("uno", "fusca")] * 2 + [("uno",)] * 20 + [("fusca",)] * 20  # so 2 pares
    df = pd.DataFrame({"veiculos": forte + fraco})
    viz = vizinhos_no_catalogo(df)
    assert viz["gol"] == {"parati": 5} and viz["parati"] == {"gol": 5}
    assert "uno" not in viz and "fusca" not in viz


def test_vizinhos_nao_e_transitivo():
    df = pd.DataFrame({"veiculos": [("gol", "parati")] * 5 + [("parati", "saveiro")] * 5})
    viz = vizinhos_no_catalogo(df)
    assert "saveiro" not in viz["gol"] and "gol" not in viz["saveiro"]
