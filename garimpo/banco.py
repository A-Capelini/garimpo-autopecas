"""Persistência SQLite: tabela pecas + índice de texto completo FTS5."""

from __future__ import annotations

import sqlite3

import pandas as pd

_SCHEMA = """
create table if not exists pecas (
    codigo text primary key,
    gtin text,
    descricao text not null,
    preco integer,
    ncm text,
    fornecedor text not null,
    texto_busca text not null
);

create index if not exists idx_pecas_gtin on pecas(gtin);
create index if not exists idx_pecas_fornecedor on pecas(fornecedor);
create index if not exists idx_pecas_ncm on pecas(ncm);

create virtual table if not exists pecas_fts using fts5(
    codigo unindexed,
    texto_busca,
    content='pecas',
    content_rowid='rowid'
);

create trigger if not exists pecas_ai after insert on pecas begin
    insert into pecas_fts(rowid, codigo, texto_busca)
    values (new.rowid, new.codigo, new.texto_busca);
end;
"""


def conectar(caminho_db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(caminho_db)
    conn.execute("pragma journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def gravar_catalogo(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Substitui o conteúdo da tabela pecas pelo DataFrame (idempotente)."""
    conn.execute("delete from pecas_fts")
    conn.execute("delete from pecas")
    df[["codigo", "gtin", "descricao", "preco", "ncm", "fornecedor", "texto_busca"]].to_sql(
        "pecas", conn, if_exists="append", index=False
    )
    conn.commit()


def buscar_fts(conn: sqlite3.Connection, consulta: str, k: int = 10) -> list[str]:
    """Busca textual FTS5 (bm25 nativo do SQLite). Devolve lista de codigo."""
    linhas = conn.execute(
        "select codigo from pecas_fts where texto_busca match ? order by bm25(pecas_fts) limit ?",
        (consulta, k),
    ).fetchall()
    return [r[0] for r in linhas]
