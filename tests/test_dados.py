from garimpo.dados import gtin_valido, normalizar_texto


def test_gtin_valido_aceita_checksum_correto():
    assert gtin_valido("7898110832573") is True


def test_gtin_valido_rejeita_checksum_errado():
    assert gtin_valido("7898110832570") is False


def test_gtin_valido_rejeita_texto_sem_gtin():
    assert gtin_valido(None) is False


def test_gtin_valido_rejeita_tamanho_invalido():
    assert gtin_valido("123") is False


def test_normalizar_texto_remove_acento_e_pontuacao():
    assert normalizar_texto("CB.CAPO UNO/PRÊMIO") == "cb capo uno premio"


def test_normalizar_texto_vazio():
    assert normalizar_texto("") == ""
