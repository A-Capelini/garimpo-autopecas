"""Interface Streamlit do Garimpo. Uso: streamlit run app.py"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from garimpo.busca import BuscadorHibrido, equivalentes_por_gtin
from garimpo.recuperacao.embeddings import OllamaEmbedder

DADOS = Path(__file__).resolve().parent / "data"
DB = DADOS / "garimpo.db"
CACHE = DADOS / "embeddings_cache.db"
EXEMPLOS = ["cabo capo uno", "palheta s435", "filtro oleo gol", "amortecedor dianteiro fusca"]

st.set_page_config(page_title="Garimpo", page_icon="🔎", layout="wide")


@st.cache_resource
def carregar():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("select * from pecas", conn)
    conn.close()
    df["categoria"] = df["ncm"].str[:4]

    # ponytail: uma conexão compartilhada entre sessões; ok pra demo de 1 usuário
    embedder = OllamaEmbedder(
        modelo="bge-m3", conn_cache=sqlite3.connect(CACHE, check_same_thread=False)
    )
    try:
        matriz = embedder.embed_lote(df["texto_busca"].tolist())
    except Exception:
        matriz = None
    return df, BuscadorHibrido(df, matriz, embedder), equivalentes_por_gtin(df)


@st.cache_data(ttl=30)
def ollama_online() -> bool:
    """Checa o servidor e já deixa o bge-m3 carregado na RAM (evita 2s na 1ª busca)."""
    try:
        requests.post(
            "http://127.0.0.1:11434/api/embed",
            json={"model": "bge-m3", "input": ["aquecimento"], "keep_alive": "30m"},
            timeout=15,
        ).raise_for_status()
        return True
    except Exception:
        return False


df, buscador, equivalentes = carregar()

# ---------- barra lateral ----------
with st.sidebar:
    st.header("Garimpo")
    st.caption("Busca híbrida offline · IA local · sem API externa")
    online = ollama_online()
    st.metric("Itens no catálogo", f"{len(df):,}".replace(",", "."))
    st.metric("Fornecedores", df["fornecedor"].nunique())
    st.write("Motor semântico (Ollama bge-m3):", "🟢 online" if online else "🔴 offline — usando só lexical")

    st.divider()
    st.subheader("Filtros")
    fornecedores = st.multiselect("Fornecedor", sorted(df["fornecedor"].unique()))
    contagem_cat = df["categoria"].value_counts()
    categorias = st.multiselect(
        "Categoria (NCM, 4 dígitos)",
        contagem_cat.index.tolist(),
        format_func=lambda c: f"{c} ({contagem_cat[c]} itens)",
    )
    preco_max = int(df["preco"].max())
    faixa_preco = st.slider("Preço", 0, preco_max, (0, preco_max))
    n_resultados = st.slider("Resultados", 5, 50, 20)

# ---------- busca ----------
st.title("🔎 Garimpo")
st.write("Encontre peças em 13 fornecedores, mesmo com descrições diferentes para o mesmo produto.")

st.text_input("O que você procura?", key="consulta", placeholder="ex.: cabo capo uno, palheta s435")
colunas = st.columns(len(EXEMPLOS))
for coluna, exemplo in zip(colunas, EXEMPLOS):
    coluna.button(exemplo, on_click=lambda e=exemplo: st.session_state.update(consulta=e))

consulta = st.session_state.get("consulta", "").strip()
if not consulta:
    st.info("Digite uma peça ou clique num exemplo acima.")
    st.stop()

inicio = time.perf_counter()
codigos, modo = buscador.buscar(consulta, k=200, usar_semantica=online)
latencia_ms = (time.perf_counter() - inicio) * 1000

resultado = df.set_index("codigo").loc[codigos].reset_index()
if fornecedores:
    resultado = resultado[resultado["fornecedor"].isin(fornecedores)]
if categorias:
    resultado = resultado[resultado["categoria"].isin(categorias)]
resultado = resultado[resultado["preco"].between(*faixa_preco)].head(n_resultados)

c1, c2, c3 = st.columns(3)
c1.metric("Modo de busca", modo)
c2.metric("Tempo de resposta", f"{latencia_ms:.0f} ms")
c3.metric("Resultados", len(resultado))

if resultado.empty:
    st.warning("Nenhuma peça encontrada com esses filtros.")
    st.stop()

resultado["equivalentes"] = resultado["gtin"].map(
    lambda g: len(equivalentes[g]) - 1 if g in equivalentes else 0
)
st.dataframe(
    resultado[["codigo", "descricao", "fornecedor", "preco", "ncm", "gtin", "equivalentes"]].rename(
        columns={
            "codigo": "Código", "descricao": "Descrição", "fornecedor": "Fornecedor",
            "preco": "Preço (R$)", "ncm": "NCM", "gtin": "GTIN",
            "equivalentes": "Em outros fornecedores",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# ---------- mesmo produto em outros fornecedores ----------
gtins = [g for g in resultado["gtin"].dropna().unique() if g in equivalentes]
if gtins:
    st.subheader("Mesmo produto em outros fornecedores (mesmo GTIN)")
    for gtin in gtins:
        grupo = equivalentes[gtin].sort_values("preco")
        mais_barato, mais_caro = grupo["preco"].min(), grupo["preco"].max()
        diferenca = (mais_caro - mais_barato) / mais_barato if mais_barato else 0
        # "\$" evita que o Streamlit trate o trecho entre dois $ como LaTeX
        titulo = f"GTIN {gtin} · {len(grupo)} fornecedores · R\\$ {mais_barato} a R\\$ {mais_caro}"
        if diferenca:
            titulo += f" ({diferenca:.0%} de diferença)"
        with st.expander(titulo):
            st.dataframe(
                grupo[["fornecedor", "codigo", "descricao", "preco"]].rename(
                    columns={"fornecedor": "Fornecedor", "codigo": "Código",
                             "descricao": "Descrição", "preco": "Preço (R$)"}
                ),
                hide_index=True,
                use_container_width=True,
            )

st.caption("Preços e fornecedores vêm sempre do banco local. Nenhum dado sai deste computador.")
