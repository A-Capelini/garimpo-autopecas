from garimpo.banco import buscar_fts, conectar, gravar_catalogo


def test_gravar_e_buscar_fts(df_pequeno):
    conn = conectar(":memory:")
    gravar_catalogo(conn, df_pequeno)
    resultados = buscar_fts(conn, "capo uno", k=5)
    assert "3502" in resultados


def test_gravar_catalogo_idempotente(df_pequeno):
    conn = conectar(":memory:")
    gravar_catalogo(conn, df_pequeno)
    gravar_catalogo(conn, df_pequeno)  # rodar 2x não deve duplicar
    total = conn.execute("select count(*) from pecas").fetchone()[0]
    assert total == len(df_pequeno)
