from garimpo.avaliacao import avaliar, construir_gabarito, hit_at_k, reciprocal_rank


def test_gabarito_agrupa_por_gtin_entre_fornecedores(df_pequeno):
    gabarito = construir_gabarito(df_pequeno)
    assert gabarito["3502"] == {"9001", "9002"}
    assert gabarito["9001"] == {"3502", "9002"}


def test_gabarito_ignora_gtin_unico(df_pequeno):
    gabarito = construir_gabarito(df_pequeno)
    assert "4001" not in gabarito
    assert "5001" not in gabarito


def test_hit_at_k_verdadeiro_quando_relevante_presente():
    assert hit_at_k(["a", "b", "c"], {"b"}) is True


def test_hit_at_k_falso_quando_ausente():
    assert hit_at_k(["a", "b"], {"z"}) is False


def test_reciprocal_rank_posicao_correta():
    assert reciprocal_rank(["x", "y", "z"], {"z"}) == 1 / 3


def test_avaliar_busca_perfeita_da_metricas_maximas(df_pequeno):
    gabarito = construir_gabarito(df_pequeno)

    def busca_perfeita(consulta: str) -> list[str]:
        return sorted(gabarito[consulta])

    m = avaliar(gabarito, busca_perfeita)
    assert m["hit@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["n_consultas"] == 3
