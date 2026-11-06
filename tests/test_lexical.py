from garimpo.recuperacao.lexical import BuscaLexical


def test_indexar_e_buscar_encontra_texto_identico(df_pequeno):
    indice = BuscaLexical()
    indice.indexar(df_pequeno["codigo"].tolist(), df_pequeno["texto_busca"].tolist())
    resultados = indice.buscar("cabo capo s alav uno", k=3)
    assert "3502" in resultados


def test_buscar_tolera_abreviacao(df_pequeno):
    indice = BuscaLexical()
    indice.indexar(df_pequeno["codigo"].tolist(), df_pequeno["texto_busca"].tolist())
    # "cb capo" é abreviação de "cabo capo" — char n-gram deve aproximar
    resultados = indice.buscar("cb capo uno", k=3)
    assert set(resultados) & {"3502", "9001", "9002"}


def test_buscar_sem_indexar_lanca_erro():
    indice = BuscaLexical()
    try:
        indice.buscar("qualquer")
        assert False, "deveria ter lançado RuntimeError"
    except RuntimeError:
        pass
