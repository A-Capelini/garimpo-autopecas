import numpy as np
import pandas as pd

from autopecas.config import RESULTADOS  # noqa: F401  (garante que o pacote importa)
from autopecas.store.medicao import formatar_markdown, medir_atualizacao
from helpers import _Sessao, linha


def catalogo_sintetico(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [linha("ABCDEFGHIJKLN"[i % 13], str(1000 + i), f"peca numero {i} modelo {i * 7}") for i in range(n)]
    )


def test_medicao_so_embeda_os_textos_novos():
    df = catalogo_sintetico(300)
    sessao = _Sessao()
    m = medir_atualizacao(df, n_novos=20, semente=1, sessao=sessao)
    assert m["itens_base"] == 280 and m["itens_novos"] == 20
    assert m["textos_novos_distintos"] == 20
    # a base inteira foi embedada na indexacao inicial (cache vazio) ...
    assert m["chamadas_http_base"] >= 1
    # ... e a atualizacao calculou exatamente os 20 textos novos, nada alem disso
    assert m["vetores_calculados"] == 20
    pedidos_da_atualizacao = [t for p in sessao.pedidos[m["chamadas_http_base"]:] for t in p]
    assert len(pedidos_da_atualizacao) == 20
    assert "inseridos 20" in m["relatorio"]


def test_medicao_com_cache_de_origem_nao_reembeda_a_base(tmp_path):
    df = catalogo_sintetico(200)
    # 1a rodada gera um embeddings.db "da Etapa 1" com TODOS os textos
    from autopecas.retrieval.ollama_embed import OllamaEmbeddingRetriever
    antigo = OllamaEmbeddingRetriever("bge-m3", cache_dir=tmp_path, sessao=_Sessao())
    antigo.fit(df["texto_busca"].tolist())
    antigo.fechar()

    sessao = _Sessao()
    m = medir_atualizacao(df, n_novos=10, semente=3, sessao=sessao, cache_origem=tmp_path / "embeddings.db")
    assert m["chamadas_http_base"] == 0  # base 100% do cache
    assert m["vetores_removidos_do_cache"] == 10  # os novos foram retirados para medir de verdade
    assert m["vetores_calculados"] == 10


def test_formatar_markdown_tem_as_etapas():
    df = catalogo_sintetico(60)
    m = medir_atualizacao(df, n_novos=5, sessao=_Sessao())
    texto = formatar_markdown(m, n_total=len(df))
    assert "atualizacao total" in texto and "embedar so os textos novos" in texto
