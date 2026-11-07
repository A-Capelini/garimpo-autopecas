"""Interface Streamlit do Garimpo. Uso: streamlit run app/Home.py

So interface: toda a logica de busca esta em src/autopecas (motor, sem Streamlit).
"""
from __future__ import annotations

import streamlit as st

from autopecas.search import Filtros
from comum import atualizar_status, aviso_catalogo_vazio, carregar

EXEMPLOS = ["cabo capo uno", "palheta s435", "filtro oleo gol", "amortecedor dianteiro fusca"]

st.set_page_config(page_title="Garimpo", page_icon="🔎", layout="wide")

itens, buscador, itens_por_gtin = carregar()
st.title("🔎 Garimpo")
if buscador is None:
    aviso_catalogo_vazio()
    st.stop()

atualizar_status(buscador)  # sonda o Ollama (1x a cada 2 s): lexical se parou, hibrido se voltou

# ---------- barra lateral ----------
with st.sidebar:
    st.header("Garimpo")
    st.caption("Busca híbrida offline · IA local · sem API externa")
    st.metric("Itens no catálogo", f"{len(itens):,}".replace(",", "."))
    st.metric("Fornecedores", itens["fornecedor"].nunique())
    if buscador.degradado:
        st.write("Motor semântico (Ollama bge-m3): 🔴 offline — usando só lexical")
    else:
        st.write("Motor semântico (Ollama bge-m3): 🟢 online")

    st.divider()
    st.subheader("Filtros")
    fornecedores = st.multiselect("Fornecedor", sorted(itens["fornecedor"].unique()))
    categorias_contagem = itens["ncm"].fillna("").str[:4].value_counts()
    categorias_contagem = categorias_contagem[categorias_contagem.index != ""]
    categorias = st.multiselect(
        "Categoria (NCM, 4 dígitos)",
        categorias_contagem.index.tolist(),
        format_func=lambda c: f"{c} ({categorias_contagem[c]} itens)",
    )
    preco_total = (float(itens["preco"].min()), float(itens["preco"].max()))
    faixa = st.slider("Preço (R$)", preco_total[0], preco_total[1], preco_total)
    n_resultados = st.slider("Resultados", 5, 50, 20)

st.write(f"Encontre peças em {itens['fornecedor'].nunique()} fornecedores, mesmo com descrições diferentes para o mesmo produto.")

# ---------- busca ----------
st.text_input("O que você procura?", key="consulta", placeholder="ex.: cabo capo uno, palheta s435")
for coluna, exemplo in zip(st.columns(len(EXEMPLOS)), EXEMPLOS):
    coluna.button(exemplo, on_click=lambda e=exemplo: st.session_state.update(consulta=e))

consulta = st.session_state.get("consulta", "").strip()
st.session_state["_ultima_consulta"] = consulta  # a página Adaptações lê daqui (o estado do widget some ao trocar de página)
if not consulta:
    st.info("Digite uma peça ou clique num exemplo acima.")
    st.stop()

filtros = Filtros(
    fornecedores=tuple(fornecedores),
    categorias_ncm=tuple(categorias),
    # o filtro de preco so entra quando o usuario mexeu no slider (evita excluir itens sem preco)
    preco_min=faixa[0] if faixa[0] > preco_total[0] else None,
    preco_max=faixa[1] if faixa[1] < preco_total[1] else None,
)
resultado = buscador.buscar(consulta, k=n_resultados, filtros=filtros)

c1, c2, c3 = st.columns(3)
c1.metric("Modo de busca", resultado.modo)
c2.metric("Tempo de resposta", f"{resultado.latencia_ms:.0f} ms")
c3.metric("Resultados", len(resultado))

if len(resultado) == 0:
    st.warning("Nenhuma peça encontrada com esses termos e filtros.")
    st.stop()

lida = resultado.consulta
tem_veiculo = lida is not None and lida.tem_veiculo
if tem_veiculo:
    ano_txt = f", ano {lida.ano}" if lida.ano else ""
    st.caption(
        f"Veículo lido da consulta: {', '.join(lida.veiculos)}{ano_txt}. Itens que citam o veículo e a peça vêm primeiro; "
        "os demais seguem a relevância e mostram na coluna **Veículo** o que a descrição diz. "
        "Para ver sugestões de adaptação (sempre a validar), abra **Adaptações** no menu à esquerda."
    )

tabela = resultado.itens.copy()
tabela["equivalentes"] = tabela["gtin_chave"].map(
    lambda g: len(itens_por_gtin[g]) - 1 if g in itens_por_gtin else 0
)
colunas = ["codigo_interno", "descricao", "fornecedor", "preco", "ncm", "gtin", "equivalentes"]
if tem_veiculo:
    colunas.insert(2, "situacao_veiculo")
st.dataframe(
    tabela[colunas].rename(
        columns={
            "codigo_interno": "Código", "descricao": "Descrição", "fornecedor": "Fornecedor",
            "preco": "Preço (R$)", "ncm": "NCM", "gtin": "GTIN", "equivalentes": "Em outros fornecedores",
            "situacao_veiculo": "Veículo",
        }
    ),
    hide_index=True,
    column_config={"Preço (R$)": st.column_config.NumberColumn(format="%.2f")},
)

# ---------- mesmo produto em outros fornecedores ----------
gtins = [g for g in tabela["gtin_chave"].dropna().unique() if g in itens_por_gtin]
if gtins:
    st.subheader("Mesmo produto em outros fornecedores (mesmo GTIN)")
    st.caption("Sugestão a validar: o mesmo GTIN indica o mesmo produto, mas confirme a aplicação no veículo.")
    for gtin in gtins:
        grupo = itens_por_gtin[gtin]
        barato, caro = grupo["preco"].min(), grupo["preco"].max()
        diferenca = (caro - barato) / barato if barato else 0
        # "\$" evita que o Streamlit trate o trecho entre dois $ como LaTeX
        titulo = f"GTIN {gtin} · {len(grupo)} fornecedores · R\\$ {barato:.2f} a R\\$ {caro:.2f}"
        if diferenca:
            titulo += f" ({diferenca:.0%} de diferença)"
        with st.expander(titulo):
            st.dataframe(
                grupo[["fornecedor", "codigo_interno", "descricao", "preco"]].rename(
                    columns={"fornecedor": "Fornecedor", "codigo_interno": "Código",
                             "descricao": "Descrição", "preco": "Preço (R$)"}
                ),
                hide_index=True,
                column_config={"Preço (R$)": st.column_config.NumberColumn(format="%.2f")},
            )

st.caption("Preços e fornecedores vêm sempre do banco local. Nenhum dado sai deste computador.")
