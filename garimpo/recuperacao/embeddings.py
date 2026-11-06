"""Cliente de embeddings locais via Ollama, com cache em SQLite.

`transporte` é injetável (protocolo com .post(url, json) -> dict) para testar
sem servidor Ollama real, seguindo o "servidor simulado" citado no relato.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Protocol

import numpy as np
import requests

_CACHE_SCHEMA = """
create table if not exists embeddings_cache (
    modelo text not null,
    texto_hash text not null,
    vetor blob not null,
    primary key (modelo, texto_hash)
);
"""


class Transporte(Protocol):
    def post(self, url: str, json: dict) -> dict: ...


class _TransporteRequests:
    def post(self, url: str, json: dict) -> dict:
        resposta = requests.post(url, json=json, timeout=30)
        resposta.raise_for_status()
        return resposta.json()


def _hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


class OllamaEmbedder:
    def __init__(
        self,
        modelo: str = "all-minilm",
        # 127.0.0.1, não localhost: no Windows localhost tenta IPv6 primeiro e
        # o Ollama só escuta IPv4 — custava ~2s de timeout por requisição.
        base_url: str = "http://127.0.0.1:11434",
        conn_cache: sqlite3.Connection | None = None,
        transporte: Transporte | None = None,
    ) -> None:
        self.modelo = modelo
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._conn = conn_cache or sqlite3.connect(":memory:")
        self._conn.executescript(_CACHE_SCHEMA)
        self._transporte = transporte or _TransporteRequests()

    def embed(self, texto: str) -> np.ndarray:
        return self.embed_lote([texto])[0]

    def embed_lote(self, textos: list[str], tamanho_lote: int = 64) -> np.ndarray:
        """Busca no cache; o que faltar vai ao Ollama em lotes via /api/embed."""
        hashes = [_hash_texto(t) for t in textos]
        vetores: dict[str, np.ndarray] = {}
        for h in set(hashes):
            linha = self._conn.execute(
                "select vetor from embeddings_cache where modelo=? and texto_hash=?",
                (self.modelo, h),
            ).fetchone()
            if linha is not None:
                vetores[h] = np.frombuffer(linha[0], dtype=np.float32)

        faltantes = list({h: t for h, t in zip(hashes, textos) if h not in vetores}.items())
        for i in range(0, len(faltantes), tamanho_lote):
            lote = faltantes[i : i + tamanho_lote]
            resposta = self._transporte.post(
                self._url, {"model": self.modelo, "input": [t for _, t in lote]}
            )
            for (h, _), emb in zip(lote, resposta["embeddings"]):
                vetor = np.array(emb, dtype=np.float32)
                vetores[h] = vetor
                self._conn.execute(
                    "insert or replace into embeddings_cache(modelo, texto_hash, vetor) values (?, ?, ?)",
                    (self.modelo, h, vetor.tobytes()),
                )
            self._conn.commit()

        return np.stack([vetores[h] for h in hashes])
