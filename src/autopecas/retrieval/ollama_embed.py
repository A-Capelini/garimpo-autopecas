"""Recuperador semantico via embeddings do Ollama (endpoint /api/embed).

Cache por texto em SQLite (padrao: data/processed/embeddings.db; com `banco=` o cache vive no
mesmo arquivo do catalogo, data/catalogo.db): a chave e (modelo,
hash do texto efetivamente enviado, ja com prefixo). Trocar 1 descricao
reembeda so 1 texto; reindexar outro nivel reaproveita tudo.

Host padrao: 127.0.0.1 (no Windows, 'localhost' pode tentar IPv6 primeiro e
estourar timeout). OLLAMA_HOST, se definido, tem prioridade.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import requests

from autopecas.config import DADOS_PROCESSADOS

NOME_BANCO = "embeddings.db"

# DDL unico da tabela de vetores: o store do catalogo (store/catalogo.py) reutiliza esta
# constante para que o cache possa viver no mesmo arquivo SQLite do catalogo.
DDL_CACHE = (
    "CREATE TABLE IF NOT EXISTS embeddings_cache ("
    "modelo TEXT NOT NULL, texto_hash TEXT NOT NULL, vetor BLOB NOT NULL, "
    "PRIMARY KEY (modelo, texto_hash))"
)


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


def hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


_hash = hash_texto  # nome antigo, mantido por compatibilidade


class OllamaEmbeddingRetriever:
    # O cosseno entre dois vetores nao depende do resto do corpus: indexa-se o catalogo
    # inteiro uma vez e a avaliacao por nivel apenas fatia.
    corpus_dependente = False

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
        banco: Path | str | None = None,
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
        # `banco` (arquivo SQLite completo) tem prioridade sobre cache_dir/embeddings.db: permite
        # guardar os vetores no mesmo arquivo do catalogo (decisao 28).
        self.caminho_banco = Path(banco) if banco is not None else self.cache_dir / NOME_BANCO
        self.usar_cache = usar_cache
        self.timeout = timeout
        self.sessao = sessao or requests.Session()
        self.nome = f"ollama:{modelo}"
        self._E: np.ndarray | None = None
        self._conn: sqlite3.Connection | None = None
        self._trava = threading.RLock()  # o app Streamlit pode chamar de threads diferentes
        self.chamadas_http = 0

    # ---------- cache ----------
    def _banco(self) -> sqlite3.Connection:
        if self._conn is None:
            self.caminho_banco.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.caminho_banco, check_same_thread=False)
            self._conn.execute(DDL_CACHE)
            self._conn.commit()
        return self._conn

    def hash_documento(self, texto: str) -> str:
        """Hash da chave de cache de um documento (texto ja com o prefixo de documento)."""
        return hash_texto(self.prefixo_documento + texto)

    def _ler_cache(self, hashes: list[str]) -> dict[str, np.ndarray]:
        achados: dict[str, np.ndarray] = {}
        with self._trava:
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
        with self._trava:
            conn = self._banco()
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings_cache (modelo, texto_hash, vetor) VALUES (?, ?, ?)",
                [(self.modelo, h, v.astype(np.float32).tobytes()) for h, v in pares],
            )
            conn.commit()

    # ---------- sonda ----------
    def disponivel(self, timeout: float = 2.0) -> bool:
        """O servidor responde e o modelo esta instalado? Sonda direta (GET /api/tags).

        Nao usa o cache de vetores: com todos os vetores em cache, fit() e consultas repetidas
        funcionam mesmo com o Ollama parado, entao so uma sonda diz a verdade sobre o servidor.
        """
        try:
            resp = self.sessao.get(f"{self.host}/api/tags", timeout=timeout)
            if resp.status_code != 200:
                return False
            nomes = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:  # sonda de saude: qualquer falha significa "indisponivel"
            return False
        base = self.modelo.split(":")[0]
        return any(n == self.modelo or n.split(":")[0] == base for n in nomes)

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
    def _embed(self, textos: Sequence[str], progresso: bool = False) -> np.ndarray:
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
        t0 = time.time()
        if progresso and itens:
            print(f"[{self.nome}] {len(achados)} em cache, {len(itens)} a calcular", file=sys.stderr)
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
            if progresso and (i // self.lote) % 50 == 0:
                feito = min(i + self.lote, len(itens))
                print(f"[{self.nome}] {feito}/{len(itens)} ({time.time() - t0:.0f}s)", file=sys.stderr)
        return np.vstack([achados[h] for h in hashes]).astype(np.float32)

    # ---------- contrato Retriever ----------
    def fit(self, textos: Sequence[str]) -> "OllamaEmbeddingRetriever":
        # float64 nos scores: reduz o ruido numerico que desfaz empates entre vetores identicos
        self._E = self._embed([self.prefixo_documento + t for t in textos], progresso=True).astype(np.float64)
        return self

    def score_rows(self, linhas: Sequence[int]) -> np.ndarray:
        assert self._E is not None, "chame fit() antes"
        return self._E[list(linhas)] @ self._E.T

    def score_texts(self, consultas: Sequence[str]) -> np.ndarray:
        assert self._E is not None, "chame fit() antes"
        Q = self._embed([self.prefixo_consulta + c for c in consultas]).astype(np.float64)
        return Q @ self._E.T

    def fechar(self) -> None:
        with self._trava:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
