"""Catalogo SQLite: fonte unica das pecas e, no mesmo arquivo, dos vetores de embedding.

Decisoes (CONTEXTO, secao 6): chave (fornecedor, codigo_interno) (19); SQLite como catalogo
unico, sem FTS5 (1); vetores no mesmo arquivo, chaveados por modelo + hash do texto (28).

Atualizacao (upsert): `importar` compara a entrada com o que ja esta gravado e so escreve o que
mudou. Os vetores nao pertencem a uma linha, e sim a um texto (hash): item novo ou com
`texto_busca` alterado gera hash novo (so ele e embedado); preco, NCM etc. nao exigem vetor.

Itens que somem da planilha nao sao apagados: com `sincronizar=True` viram `ativo = 0`
(historico preservado, reativam se voltarem). A busca deve usar so os ativos.

Conexao: uma por instancia, para uso em um unico thread (o app abre, le e fecha).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from autopecas.config import CATALOGO_DB
from autopecas.retrieval.ollama_embed import DDL_CACHE

VERSAO_ESQUEMA = 1

COLUNAS_ENTRADA = (
    "codigo_interno", "gtin", "gtin_status", "gtin_chave",
    "descricao", "preco", "ncm", "fornecedor", "texto_busca",
)
# Campos comparados para decidir se o item mudou (a chave nao entra).
_PAYLOAD = ("gtin", "gtin_status", "gtin_chave", "descricao", "preco", "ncm", "texto_busca")

_DDL_ITENS = """
CREATE TABLE IF NOT EXISTS itens (
    fornecedor     TEXT NOT NULL,
    codigo_interno TEXT NOT NULL,
    gtin           TEXT,
    gtin_status    TEXT,
    gtin_chave     TEXT,
    descricao      TEXT NOT NULL,
    preco          REAL,
    ncm            TEXT,
    texto_busca    TEXT NOT NULL,
    ativo          INTEGER NOT NULL DEFAULT 1 CHECK (ativo IN (0, 1)),
    criado_em      TEXT NOT NULL,
    atualizado_em  TEXT NOT NULL,
    PRIMARY KEY (fornecedor, codigo_interno)
);
CREATE INDEX IF NOT EXISTS idx_itens_gtin_chave ON itens (gtin_chave);
CREATE INDEX IF NOT EXISTS idx_itens_ncm ON itens (ncm);
"""


@dataclass(frozen=True)
class RelatorioImportacao:
    inseridos: int
    atualizados: int
    inalterados: int
    reativados: int
    desativados: int
    precos_alterados: int
    textos_novos: int  # textos de busca ineditos no catalogo = vetores que talvez precisem ser calculados

    def resumo(self) -> str:
        return (
            f"inseridos {self.inseridos} | atualizados {self.atualizados} "
            f"(precos alterados {self.precos_alterados}) | inalterados {self.inalterados} | "
            f"reativados {self.reativados} | desativados {self.desativados} | "
            f"textos de busca novos {self.textos_novos}"
        )


def _agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _py(valor):
    """Valor pandas -> valor Python aceito pelo sqlite (ausente vira None)."""
    if valor is None or valor is pd.NA or (not isinstance(valor, str) and pd.isna(valor)):
        return None
    if hasattr(valor, "item"):  # escalar numpy
        return valor.item()
    return valor


class Catalogo:
    def __init__(self, caminho: Path | str = CATALOGO_DB, agora: Callable[[], str] = _agora_utc) -> None:
        self.caminho = str(caminho)
        self._agora = agora
        if self.caminho != ":memory:":
            Path(self.caminho).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.caminho)
        versao = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if versao > VERSAO_ESQUEMA:
            self._conn.close()
            raise RuntimeError(
                f"{self.caminho} tem esquema versao {versao}, mais novo que o suportado ({VERSAO_ESQUEMA}). "
                "Atualize o codigo do Garimpo."
            )
        if self.caminho != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL_ITENS)
        self._conn.execute(DDL_CACHE)
        self._conn.execute(f"PRAGMA user_version = {VERSAO_ESQUEMA}")
        self._conn.commit()

    # ---------- ciclo de vida ----------
    def fechar(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Catalogo":
        return self

    def __exit__(self, *_exc) -> None:
        self.fechar()

    # ---------- leitura ----------
    def itens(self, apenas_ativos: bool = True) -> pd.DataFrame:
        """DataFrame na ordem de insercao, com as colunas do catalogo.csv + ativo/criado_em/atualizado_em."""
        where = "WHERE ativo = 1" if apenas_ativos else ""
        df = pd.read_sql_query(
            "SELECT codigo_interno, gtin, gtin_status, gtin_chave, descricao, preco, ncm, fornecedor, "
            f"texto_busca, ativo, criado_em, atualizado_em FROM itens {where} ORDER BY rowid",
            self._conn,
        )
        df["ativo"] = df["ativo"].astype(bool)
        return df

    def contar(self) -> dict[str, int]:
        ativos, inativos = self._conn.execute(
            "SELECT COALESCE(SUM(ativo = 1), 0), COALESCE(SUM(ativo = 0), 0) FROM itens"
        ).fetchone()
        return {"ativos": int(ativos), "inativos": int(inativos)}

    def contar_vetores(self, modelo: str | None = None) -> int:
        if modelo is None:
            return self._conn.execute("SELECT COUNT(*) FROM embeddings_cache").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM embeddings_cache WHERE modelo = ?", (modelo,)
        ).fetchone()[0]

    # ---------- escrita ----------
    def importar(
        self, df: pd.DataFrame, sincronizar: bool = False, simular: bool = False
    ) -> RelatorioImportacao:
        """Upsert por (fornecedor, codigo_interno). Tudo numa transacao: falhou, nada muda.

        sincronizar=True: itens ativos dos fornecedores presentes em `df` que nao estao em `df`
        passam a inativos (a planilha e tratada como o estoque completo desses fornecedores).
        Fornecedores ausentes de `df` nao sao tocados.

        simular=True: calcula e devolve o relatorio sem gravar nada (pre-visualizacao no app).
        """
        self._validar(df)
        agora = self._agora()

        entrada: dict[tuple[str, str], tuple] = {}
        for linha in df.loc[:, list(COLUNAS_ENTRADA)].itertuples(index=False):
            reg = dict(zip(COLUNAS_ENTRADA, (_py(v) for v in linha)))
            chave = (reg["fornecedor"], reg["codigo_interno"])
            entrada[chave] = tuple(reg[c] for c in _PAYLOAD)

        existentes: dict[tuple[str, str], tuple] = {}
        textos_existentes: set[str] = set()
        cols = ", ".join(_PAYLOAD)
        for fornecedor, codigo, ativo, *payload in self._conn.execute(
            f"SELECT fornecedor, codigo_interno, ativo, {cols} FROM itens"
        ):
            existentes[(fornecedor, codigo)] = (ativo, tuple(payload))
            textos_existentes.add(payload[_PAYLOAD.index("texto_busca")])

        i_preco, i_texto = _PAYLOAD.index("preco"), _PAYLOAD.index("texto_busca")
        inserir, atualizar = [], []
        inalterados = reativados = mudados = precos = 0
        textos_novos: set[str] = set()
        for chave, payload in entrada.items():
            if chave not in existentes:
                inserir.append((*chave, *payload, agora, agora))
                textos_novos.add(payload[i_texto])
                continue
            ativo_antigo, payload_antigo = existentes[chave]
            mudou = payload != payload_antigo
            reativar = ativo_antigo == 0
            if not mudou and not reativar:
                inalterados += 1
                continue
            if reativar:
                reativados += 1
            if mudou:
                mudados += 1
                if payload[i_preco] != payload_antigo[i_preco]:
                    precos += 1
                if payload[i_texto] != payload_antigo[i_texto]:
                    textos_novos.add(payload[i_texto])
            atualizar.append((*payload, agora, *chave))

        desativar: list[tuple] = []
        if sincronizar:
            fornecedores = {f for f, _ in entrada}
            desativar = [
                (agora, f, c)
                for (f, c), (ativo, _) in existentes.items()
                if ativo == 1 and f in fornecedores and (f, c) not in entrada
            ]

        if not simular:
            self._gravar(inserir, atualizar, desativar)

        return RelatorioImportacao(
            inseridos=len(inserir),
            atualizados=mudados,
            inalterados=inalterados,
            reativados=reativados,
            desativados=len(desativar),
            precos_alterados=precos,
            textos_novos=len(textos_novos - textos_existentes),
        )

    def _gravar(self, inserir: list, atualizar: list, desativar: list) -> None:
        cols = ", ".join(_PAYLOAD)
        with self._conn:  # BEGIN ... COMMIT (ROLLBACK se der excecao)
            self._conn.executemany(
                f"INSERT INTO itens (fornecedor, codigo_interno, {cols}, ativo, criado_em, atualizado_em) "
                f"VALUES (?, ?, {', '.join('?' * len(_PAYLOAD))}, 1, ?, ?)",
                inserir,
            )
            self._conn.executemany(
                "UPDATE itens SET " + ", ".join(f"{c} = ?" for c in _PAYLOAD)
                + ", ativo = 1, atualizado_em = ? WHERE fornecedor = ? AND codigo_interno = ?",
                atualizar,
            )
            self._conn.executemany(
                "UPDATE itens SET ativo = 0, atualizado_em = ? WHERE fornecedor = ? AND codigo_interno = ?",
                desativar,
            )

    def remover_vetores(self, modelo: str, hashes: list[str]) -> int:
        """Apaga vetores do cache (por modelo e hash). Devolve quantos foram removidos."""
        removidos = 0
        with self._conn:
            for i in range(0, len(hashes), 500):  # limite de variaveis do SQLite
                parte = hashes[i : i + 500]
                cur = self._conn.execute(
                    f"DELETE FROM embeddings_cache WHERE modelo = ? AND texto_hash IN ({','.join('?' * len(parte))})",
                    [modelo, *parte],
                )
                removidos += cur.rowcount
        return removidos

    def absorver_cache(self, caminho_antigo: Path | str) -> int:
        """Copia vetores de um embeddings.db antigo (Etapa 1) para este arquivo; devolve quantos entraram.

        Evita refazer a indexacao (~10 min). Idempotente: linhas que ja existem sao ignoradas.
        """
        antes = self.contar_vetores()
        self._conn.execute("ATTACH DATABASE ? AS antigo", (str(caminho_antigo),))
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO embeddings_cache (modelo, texto_hash, vetor) "
                    "SELECT modelo, texto_hash, vetor FROM antigo.embeddings_cache"
                )
        finally:
            self._conn.execute("DETACH DATABASE antigo")
        return self.contar_vetores() - antes

    # ---------- validacao ----------
    @staticmethod
    def _validar(df: pd.DataFrame) -> None:
        faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
        if faltando:
            raise ValueError(f"colunas ausentes na entrada: {faltando}")
        for coluna in ("codigo_interno", "fornecedor", "descricao", "texto_busca"):
            n = int(df[coluna].isna().sum())
            if n:
                raise ValueError(f"{n} linha(s) com '{coluna}' ausente")
        vazios = int((df["codigo_interno"].astype(str).str.strip() == "").sum())
        if vazios:
            raise ValueError(f"{vazios} linha(s) com 'codigo_interno' vazio")
        dup = df.duplicated(["fornecedor", "codigo_interno"], keep=False)
        if dup.any():
            exemplos = df.loc[dup, ["fornecedor", "codigo_interno"]].drop_duplicates().head(5)
            raise ValueError(
                f"{int(dup.sum())} linha(s) repetem (fornecedor, codigo_interno); exemplos: "
                + "; ".join(f"{f}/{c}" for f, c in exemplos.itertuples(index=False))
            )
