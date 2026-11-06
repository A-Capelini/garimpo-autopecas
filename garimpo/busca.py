"""Buscador híbrido (lexical + semântico via RRF), independente de interface.

Configuração vencedora da comparação (scripts/comparar_embeddings.py):
bge-m3 sobre texto normalizado, RRF com peso 3:1 a favor do lexical.
Se o Ollama estiver fora do ar, cai para busca só lexical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from garimpo.dados import gtin_valido, normalizar_texto
from garimpo.recuperacao.lexical import BuscaLexical
from garimpo.recuperacao.semantica import BuscaSemantica, fusao_rrf

PROFUNDIDADE = 50
PESO_LEXICAL = 3.0


class BuscadorHibrido:
    def __init__(self, df: pd.DataFrame, matriz: np.ndarray | None, embedder=None) -> None:
        self.df = df.reset_index(drop=True)
        self._embedder = embedder
        self._lexical = BuscaLexical()
        self._lexical.indexar(self.df["codigo"].tolist(), self.df["texto_busca"].tolist())
        self._semantica = None
        if matriz is not None:
            self._semantica = BuscaSemantica()
            self._semantica.indexar(self.df["codigo"].tolist(), matriz)

    def buscar(
        self, consulta: str, k: int = PROFUNDIDADE, usar_semantica: bool = True
    ) -> tuple[list[str], str]:
        """Devolve (codigos ordenados, modo usado: 'híbrida' ou 'lexical')."""
        lexical = self._lexical.buscar(consulta, k=k)
        if not usar_semantica or self._semantica is None or self._embedder is None:
            return lexical, "lexical"
        try:
            vetor = self._embedder.embed(normalizar_texto(consulta))
        except Exception:
            return lexical, "lexical"
        semantica = self._semantica.buscar_vetor(vetor, k=k)
        return fusao_rrf([lexical, semantica], pesos=[PESO_LEXICAL, 1.0])[:k], "híbrida"


def equivalentes_por_gtin(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """GTIN válido -> linhas do catálogo com esse GTIN (só GTINs em 2+ fornecedores)."""
    validos = df[df["gtin"].apply(gtin_valido)]
    grupos = validos.groupby("gtin")
    return {g: linhas for g, linhas in grupos if linhas["fornecedor"].nunique() > 1}
