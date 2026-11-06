"""Pedacos compartilhados pelas paginas do app (cache do buscador, mensagens)."""
from __future__ import annotations

import streamlit as st

from autopecas.config import CATALOGO_DB
from autopecas.search import Buscador, equivalentes_por_gtin
from autopecas.store import Catalogo


@st.cache_resource(show_spinner="Carregando o catálogo e os vetores…")
def carregar():
    """(itens ativos, Buscador, equivalentes por GTIN); (None, None, None) se o catálogo não existe ou está vazio.

    O Buscador so chama o Ollama para textos sem vetor no cache; se o Ollama estiver fora do ar,
    abre em modo lexical (e `atualizar_status` volta ao hibrido quando o servidor reaparece).
    """
    if not CATALOGO_DB.exists():
        return None, None, None
    with Catalogo(CATALOGO_DB) as cat:
        itens = cat.itens()
    if itens.empty:
        return None, None, None
    buscador = Buscador.criar(itens, banco=CATALOGO_DB)
    buscador.aquecer()
    return itens, buscador, equivalentes_por_gtin(itens)  # calculado 1x, nao a cada interacao


@st.cache_data(ttl=2, show_spinner=False)
def atualizar_status(_buscador: Buscador) -> bool:
    """Sonda o Ollama (no maximo 1x a cada 2 s): cai para lexical se parou e volta se voltou."""
    return _buscador.verificar_servidor()


def aviso_catalogo_vazio() -> None:
    st.warning(
        "O catálogo ainda está vazio. Abra a página **Atualizar catálogo** (menu à esquerda) e envie "
        "a planilha, ou rode `python scripts/06_importar_catalogo.py` no terminal."
    )
