import pandas as pd

from autopecas.data.quality import (
    aplicar_qualidade,
    classificar_gtin,
    digito_verificador_ok,
    normalizar_texto,
)


def test_digito_verificador_ean13_real_e_adulterado():
    assert digito_verificador_ok("7898511031964")  # GTIN real da planilha
    assert not digito_verificador_ok("7898511031965")  # ultimo digito trocado
    assert not digito_verificador_ok("12345")  # comprimento invalido
    assert not digito_verificador_ok("78985110319AB")


def test_classificar_gtin():
    assert classificar_gtin("7898511031964", "510247") == "valido"
    assert classificar_gtin("SEM GTIN", "1") == "sem_gtin_texto"
    assert classificar_gtin("sem gtin", "1") == "sem_gtin_texto"
    assert classificar_gtin(float("nan"), "1") == "ausente"
    assert classificar_gtin("", "1") == "ausente"
    assert classificar_gtin("36206", "36206") == "igual_codigo_interno"
    assert classificar_gtin("12345", "999") == "invalido"


def test_normalizar_texto():
    assert normalizar_texto("JUNTA DA VÁLVULA TERMOSTÁTICA -:") == "junta da valvula termostatica"
    assert normalizar_texto("CB.CAPO UNO/PRE 84>") == "cb capo uno/pre 84>"
    assert normalizar_texto("  Palheta   (14\")  ") == "palheta 14"
    assert normalizar_texto(float("nan")) == ""


def test_aplicar_qualidade_cria_colunas():
    df = pd.DataFrame(
        {
            "codigo_interno": ["1", "2", "3"],
            "gtin": ["7898511031964", "SEM GTIN", "3"],
            "descricao": ["Item A", "Item B", "Item C"],
            "preco": [10, 0, 5],
            "ncm": ["1", "2", "3"],
            "fornecedor": ["A", "B", "N"],
        }
    )
    out = aplicar_qualidade(df)
    assert out["gtin_status"].tolist() == ["valido", "sem_gtin_texto", "igual_codigo_interno"]
    assert out["gtin_chave"].notna().tolist() == [True, False, False]
    assert out["preco_zero"].tolist() == [False, True, False]
    assert out["texto_busca"].tolist() == ["item a", "item b", "item c"]
