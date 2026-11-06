"""Recuperador semantico via embeddings do Ollama (endpoint /api/embed).

Cache por texto em SQLite (data/processed/embeddings.db): a chave e (modelo,
hash do texto efetivamente enviado, ja com prefixo). Trocar 1 descricao
reembeda so 1 texto; reindexar outro nivel reaproveita tudo.

Host padrao: 127.0.0.1 (no Windows, 'localhost' pode tentar IPv6 primeiro e
estourar timeout). OLLAMA_HOST, se definido, tem prioridade.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import requests

from autopecas.config import DADOS_PROCESSADOS

NOME_BANCO = "embeddings.db"


class ErroOllama(RuntimeError):
    """Falha ao falar com o Ollama (servidor fora do ar ou modelo ausente)."""


def _host_padrao() -> str:
    bruto = os.environ.get("OLLAMA_HOST", "").strip()
    if not bruto:
        return "http://127.0.0.1:11434"
    if "://" not in bruto:
        bruto = "http://" + bruto
    # 0.0.0.0 e endereco de escuta, nao de destino
    return bruto.replace("//0.0.0.0", "//127.0.0.1").rstrip("/")


def _hash(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


class OllamaEmbeddingRetriever:
    def __init__(
        self,
        modelo: str,
        host: str | None = None,
        lote: int = 32,
        keep_alive: str = "30m",
        prefixo_documento: str = "",
        prefixo_consulta: str = "",
        cache_dir: Path | str = DADOS_PROCESSADOS,
        usar_cache: bool = True,
        timeout: float = 600,
        sessao=None,
    ) -> None:
        self.modelo = modelo
        self.host = (host or _host_padrao()).rstrip("/")
        self.lote = lote
        self.keep_alive = keep_alive
        self.prefixo_documento = prefixo_documento
        self.prefixo_consulta = prefixo_consulta
        self.cache_dir = Path(cache_dir)
        self.usar_cache = usar_cache
        self.timeout = timeout
        self.sessao = sessao or requests.Session()
        self.nome = f"ollama:{modelo}"
        self._E: np.ndarray | None = None
        self._conn: sqlite3.Connection | None = None
        self.chamadas_http = 0

    # ---------- cache ----------
    def _banco(self) -> sqlite3.Connection:
        if self._conn is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.cache_dir / NOME_BANCO)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings_cache ("
                "modelo TEXT NOT NULL, texto_hash TEXT NOT NULL, vetor BLOB NOT NULL, "
                "PRIMARY KEY (modelo, texto_hash))"
            )
            self._conn.commit()
        return self._conn

    def _ler_cache(self, hashes: list[str]) -> dict[str, np.ndarray]:
        achados: dict[str, np.ndarray] = {}
        conn = self._banco()
        for i in range(0, len(hashes), 500):  # limite de variaveis do SQLite
            parte = hashes[i : i + 500]
            marcas = ",".join("?" * len(parte))
            cur = conn.execute(
                f"SELECT texto_hash, vetor FROM embeddings_cache "
                f"WHERE modelo = ? AND texto_hash IN ({marcas})",
                [self.modelo, *parte],
            )
            for h, blob in cur:
                achados[h] = np.frombuffer(blob, dtype=np.float32)
        return achados

    def _gravar_cache(self, pares: list[tuple[str, np.ndarray]]) -> None:
        conn = self._banco()
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings_cache (modelo, texto_hash, vetor) VALUES (?, ?, ?)",
            [(self.modelo, h, v.astype(np.float32).tobytes()) for h, v in pares],
        )
        conn.commit()

    # ---------- HTTP ----------
    def _post_embed(self, textos: list[str]) -> list[list[float]]:
        url = f"{self.host}/api/embed"
        corpo = {"model": self.modelo, "input": textos, "keep_alive": self.keep_alive}
        self.chamadas_http += 1
        try:
            resp = self.sessao.post(url, json=corpo, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ErroOllama(
                f"Nao consegui falar com o Ollama em {self.host}. "
                "Verifique: systemctl status ollama  (ou rode: ollama serve)"
            ) from e
        if resp.status_code == 404:
            raise ErroOllama(f"Modelo '{self.modelo}' nao encontrado. Rode: ollama pull {self.modelo}")
        if resp.status_code != 200:
            raise ErroOllama(f"Ollama respondeu HTTP {resp.status_code}: {getattr(resp, 'text', '')[:200]}")
        return resp.json()["embeddings"]

    # ---------- embeddings ----------
    def _embed(self, textos: Sequence[str]) -> np.ndarray:
        """Vetores unitarios, na ordem de entrada. Usa o cache por texto."""
        textos = list(textos)
        if not textos:
            return np.zeros((0, 0), dtype=np.float32)
        hashes = [_hash(t) for t in textos]
        achados = self._ler_cache(list(dict.fromkeys(hashes))) if self.usar_cache else {}

        faltam: dict[str, str] = {}  # hash -> texto (sem duplicatas)
        for h, t in zip(hashes, textos):
            if h not in achados and h not in faltam:
                faltam[h] = t
        itens = list(faltam.items())
        for i in range(0, len(itens), self.lote):
            parte = itens[i : i + self.lote]
            vetores = self._post_embed([t for _, t in parte])
            novos = []
            for (h, _), v in zip(parte, vetores):
                v = np.asarray(v, dtype=np.float32)
                n = np.linalg.norm(v)
                v = v / n if n > 0 else v
                achados[h] = v
                novos.append((h, v))
            if self.usar_cache:
                self._gravar_cache(novos)  # grava por lote: queda no meio nao perde tudo
        return np.vstack([achados[h] for h in hashes]).astype(np.float32)

    # ---------- contrato Retriever ----------
    def fit(self, textos: Sequence[str]) -> "OllamaEmbeddingRetriever":
        self._E = self._embed([self.prefixo_documento + t for t in textos])
        return self

    def score_rows(self, linhas: Sequence[int]) -> np.ndarray:
        assert self._E is not None, "chame fit() antes"
        return self._E[list(linhas)] @ self._E.T

    def score_texts(self, consultas: Sequence[str]) -> np.ndarray:
        assert self._E is not None, "chame fit() antes"
        Q = self._embed([self.prefixo_consulta + c for c in consultas])
        return Q @ self._E.T

    def fechar(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
