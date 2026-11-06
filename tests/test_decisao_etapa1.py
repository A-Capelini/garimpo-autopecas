import pandas as pd

from autopecas.data import quality
from autopecas.data.quality import aplicar_qualidade, normalizar_texto
from autopecas.retrieval.lexical import TfidfRetriever


def test_aplicar_qualidade_usa_a_normalizacao_decidida():
    df = pd.DataFrame([{
        "codigo_interno": "1", "gtin": "SEM GTIN", "descricao": "CB.CAPO UNO/PRE 84>",
        "preco": 10, "ncm": "87089990", "fornecedor": "C",
    }])
    assert quality.MANTER_SIMBOLOS_NO_TEXTO_BUSCA is False
    assert aplicar_qualidade(df)["texto_busca"].iloc[0] == "cb capo uno pre 84"


def test_funcao_continua_com_padrao_antigo_quando_chamada_diretamente():
    # a decisao vive na constante do pipeline; a funcao preserva o comportamento explicito
    assert normalizar_texto("CB.CAPO UNO/PRE 84>") == "cb capo uno/pre 84>"


def test_lexical_padrao_e_o_escolhido():
    assert TfidfRetriever().nome == "tfidf-char"
    assert TfidfRetriever(min_df=2).nome == "tfidf-char[min_df=2,sublinear=True]"
