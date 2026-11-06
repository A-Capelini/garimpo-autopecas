"""Buscador do Garimpo (motor, sem Streamlit): consulta em texto livre -> itens ranqueados.

Configuracao adotada na Etapa 1 (decisoes 21 a 23): texto de busca "simples", lexical TF-IDF de
caracteres (3,5) e hibrido RRF 3:1 a favor do lexical, k_rrf=1, profundidade 50, com bge-m3.

Diferencas em relacao ao app do Vitor (corrigidas aqui):
- filtros (fornecedor, NCM, preco) valem ANTES do top-k: os ranks do RRF sao calculados so entre
  os itens que passam nos filtros, em vez de cortar um top-200 ja feito;
- se o Ollama falhar (na carga ou no meio de uma consulta) o buscador degrada para lexical sem
  erro e `reconectar()` volta ao hibrido quando o servidor reaparece;
- a consulta e normalizada com a mesma regra do `texto_busca` do catalogo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autopecas.config import CATALOGO_DB
from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA, normalizar_texto
from autopecas.retrieval.hybrid import DECIMAIS_EMPATE, HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever

PESO_LEXICAL = 3.0
PESO_SEMANTICO = 1.0
K_RRF = 1
PROFUNDIDADE = 50
MODELO_EMBEDDING = "bge-m3"
MODO_HIBRIDA = "híbrida"
MODO_LEXICAL = "lexical"


@dataclass(frozen=True)
class Filtros:
    fornecedores: tuple[str, ...] = ()
    categorias_ncm: tuple[str, ...] = ()  # prefixo de NCM (o app usa 4 digitos)
    preco_min: float | None = None
    preco_max: float | None = None

    def mascara(self, itens: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(itens), dtype=bool)
        if self.fornecedores:
            m &= itens["fornecedor"].isin(self.fornecedores).to_numpy()
        if self.categorias_ncm:
            ncm = itens["ncm"].fillna("").astype(str)
            m &= np.logical_or.reduce([ncm.str.startswith(c).to_numpy() for c in self.categorias_ncm])
        preco = itens["preco"]
        if self.preco_min is not None:
            m &= (preco >= self.preco_min).fillna(False).to_numpy()
        if self.preco_max is not None:
            m &= (preco <= self.preco_max).fillna(False).to_numpy()
        return m


@dataclass(frozen=True)
class Resultado:
    itens: pd.DataFrame  # itens do catalogo + colunas `score` e `posicao`
    modo: str  # MODO_HIBRIDA ou MODO_LEXICAL
    latencia_ms: float

    def __len__(self) -> int:
        return len(self.itens)


class Buscador:
    def __init__(self, itens: pd.DataFrame, hibrido: HybridRetriever) -> None:
        """`itens`: itens ATIVOS do catalogo (Catalogo.itens()). `hibrido`: com fallback_lexical=True."""
        if not hibrido.fallback_lexical:
            raise ValueError("o Buscador exige HybridRetriever(fallback_lexical=True)")
        self.itens = itens.reset_index(drop=True)
        self.hibrido = hibrido
        self._textos = self.itens["texto_busca"].tolist()
        self.hibrido.fit(self._textos)

    @classmethod
    def criar(
        cls,
        itens: pd.DataFrame,
        banco=CATALOGO_DB,
        modelo: str = MODELO_EMBEDDING,
        host: str | None = None,
        sessao=None,
    ) -> "Buscador":
        semantico = OllamaEmbeddingRetriever(modelo, host=host, banco=banco, sessao=sessao)
        hibrido = HybridRetriever(
            TfidfRetriever(), semantico, metodo="rrf", peso_lexical=PESO_LEXICAL,
            peso_semantico=PESO_SEMANTICO, k_rrf=K_RRF, profundidade=PROFUNDIDADE,
            fallback_lexical=True,
        )
        return cls(itens, hibrido)

    # ---------- estado ----------
    @property
    def degradado(self) -> bool:
        return self.hibrido.degradado

    @property
    def modo(self) -> str:
        return MODO_LEXICAL if self.degradado else MODO_HIBRIDA

    def reconectar(self) -> bool:
        """Tenta voltar ao modo hibrido (Ollama de volta). True se ficou/esta hibrido."""
        if not self.degradado:
            return True
        try:
            self.hibrido.semantico.fit(self._textos)
        except ErroOllama:
            return False
        self.hibrido.degradado = False
        return True

    # ---------- busca ----------
    def buscar(self, consulta: str, k: int = 20, filtros: Filtros | None = None) -> Resultado:
        inicio = time.perf_counter()
        q = normalizar_texto(consulta, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA)
        mascara = filtros.mascara(self.itens) if filtros else np.ones(len(self.itens), dtype=bool)
        idx = np.flatnonzero(mascara)
        if not q or len(idx) == 0:
            return self._resultado(idx[:0], np.array([]), self.modo, inicio)

        Sl = np.asarray(self.hibrido.lexical.score_texts([q]), dtype=np.float64)[:, idx]
        modo, score = MODO_LEXICAL, Sl[0]
        if not self.degradado:
            try:
                Ss = np.asarray(self.hibrido.semantico.score_texts([q]), dtype=np.float64)[:, idx]
                score, modo = self.hibrido.fundir(Sl, Ss)[0], MODO_HIBRIDA
                # O RRF da nota a quem so empata em zero (posicao 1..50 existe mesmo sem evidencia):
                # item sem sinal em nenhuma das duas fontes nao pode aparecer.
                score = np.where((Sl[0] > 0) | (Ss[0] > 0), score, 0.0)
            except ErroOllama:
                self.hibrido.degradado = True  # o proximo reconectar() tenta de novo

        ordem = np.argsort(-np.round(score, DECIMAIS_EMPATE), kind="stable")[:k]
        ordem = ordem[score[ordem] > 0]  # score 0 = nenhuma evidencia: nao lista
        return self._resultado(idx[ordem], score[ordem], modo, inicio)

    def _resultado(self, posicoes: np.ndarray, scores: np.ndarray, modo: str, inicio: float) -> Resultado:
        itens = self.itens.iloc[posicoes].copy()
        itens["score"] = scores
        itens["posicao"] = np.arange(1, len(itens) + 1)
        return Resultado(itens.reset_index(drop=True), modo, (time.perf_counter() - inicio) * 1000)


def equivalentes_por_gtin(itens: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """GTIN valido -> itens com esse GTIN, so para GTINs presentes em 2+ fornecedores (por preco).

    Usa `gtin_chave` (GTIN valido e diferente do codigo interno). E sugestao a validar: o mesmo
    GTIN indica o mesmo produto, mas a compatibilidade com o veiculo nao e garantida.
    """
    com_chave = itens[itens["gtin_chave"].notna()]
    grupos = com_chave.groupby("gtin_chave")
    return {
        g: linhas.sort_values("preco", kind="stable").reset_index(drop=True)
        for g, linhas in grupos
        if linhas["fornecedor"].nunique() > 1
    }
