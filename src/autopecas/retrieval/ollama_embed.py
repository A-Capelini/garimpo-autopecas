"""Recuperador por embeddings gerados localmente pelo Ollama (endpoint /api/embed).

Tudo roda na maquina: nenhum dado sai do computador.
O corpus e vetorizado uma vez e guardado em cache (.npy) em data/processed/.
"""
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import requests

from autopecas.config import DADOS_PROCESSADOS


class ErroOllama(RuntimeError):
    """Erro de comunicacao com o servidor Ollama (mensagem em portugues, com a solucao)."""


def _host_padrao() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


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
        timeout: int = 600,
        sessao=None,
    ):
        self.modelo = modelo
        self.nome = f"ollama:{modelo}"
        self.host = (host or _host_padrao()).rstrip("/")
        self.lote = lote
        self.keep_alive = keep_alive
        self.prefixo_documento = prefixo_documento  # ex.: "passage: " (e5) / "search_document: " (nomic)
        self.prefixo_consulta = prefixo_consulta  # ex.: "query: " (e5) / "search_query: " (nomic)
        self.cache_dir = Path(cache_dir)
        self.usar_cache = usar_cache
        self.timeout = timeout
        self._http = sessao or requests
        self._E: np.ndarray | None = None

    # ---- comunicacao com o Ollama -------------------------------------------------
    def _post_embed(self, textos: list[str]) -> list[list[float]]:
        try:
            r = self._http.post(
                f"{self.host}/api/embed",
                json={"model": self.modelo, "input": textos, "keep_alive": self.keep_alive},
                timeout=self.timeout,
            )
        except requests.ConnectionError as e:
            raise ErroOllama(
                f"Nao consegui falar com o Ollama em {self.host}. "
                "Verifique: systemctl status ollama  (ou rode: ollama serve)"
            ) from e
        if r.status_code == 404:
            raise ErroOllama(f"Modelo '{self.modelo}' nao encontrado. Rode: ollama pull {self.modelo}")
        r.raise_for_status()
        return r.json()["embeddings"]

    def _embed(self, textos: list[str], prefixo: str, progresso: bool = False) -> np.ndarray:
        blocos = []
        t0 = time.time()
        for i in range(0, len(textos), self.lote):
            lote = [prefixo + t for t in textos[i : i + self.lote]]
            blocos.append(np.asarray(self._post_embed(lote), dtype=np.float32))
            if progresso and (i // self.lote) % 20 == 0:
                feitos = min(i + self.lote, len(textos))
                print(f"  [{self.modelo}] {feitos}/{len(textos)} textos ({time.time() - t0:.0f}s)", flush=True)
        m = np.vstack(blocos)
        norma = np.linalg.norm(m, axis=1, keepdims=True)
        norma[norma == 0] = 1.0
        return m / norma  # vetores unitarios: similaridade de cosseno = produto escalar

    # ---- cache --------------------------------------------------------------------
    def _arquivo_cache(self, textos: list[str]) -> Path:
        h = hashlib.sha1(("\n".join(textos) + "|" + self.prefixo_documento).encode("utf-8")).hexdigest()[:10]
        seguro = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.modelo)
        return self.cache_dir / f"emb_{seguro}_{h}.npy"

    # ---- API do Retriever ---------------------------------------------------------
    def fit(self, textos: Sequence[str]) -> None:
        textos = list(textos)
        arq = self._arquivo_cache(textos)
        if self.usar_cache and arq.exists():
            self._E = np.load(arq)
            return
        self._E = self._embed(textos, self.prefixo_documento, progresso=True)
        if self.usar_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(arq, self._E)

    def score_rows(self, linhas: Sequence[int]) -> np.ndarray:
        """Consulta-por-exemplo: usa o vetor de documento da propria linha (sem nova chamada)."""
        return self._E[list(linhas)] @ self._E.T

    def score_texts(self, consultas: Sequence[str]) -> np.ndarray:
        Q = self._embed(list(consultas), self.prefixo_consulta)
        return Q @ self._E.T
