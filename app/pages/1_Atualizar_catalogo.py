"""Pagina: importar a planilha de fornecedores (novos itens, precos alterados, itens que sairam)."""
from __future__ import annotations

import streamlit as st

from autopecas.config import CATALOGO_DB
from autopecas.data.loader import carregar_xlsx
from autopecas.data.quality import aplicar_qualidade
from autopecas.store import Catalogo
from comum import carregar

st.set_page_config(page_title="Atualizar catálogo · Garimpo", page_icon="🔄", layout="wide")
st.title("🔄 Atualizar catálogo")
st.write(
    "Envie a planilha consolidada (sem cabeçalho: código, GTIN, descrição, preço, NCM, fornecedor). "
    "Itens novos entram, preços alterados são atualizados e **nada é apagado**. A planilha é lida "
    "neste computador e não é enviada a lugar nenhum."
)

arquivo = st.file_uploader("Planilha (.xlsx)", type=["xlsx"])
sincronizar = st.checkbox(
    "A planilha é o estoque COMPLETO dos fornecedores que ela contém: marcar como inativos os "
    "itens deles que não estão nela",
    value=False,
)
if arquivo is None:
    st.info("Escolha um arquivo para ver o que mudaria antes de aplicar.")
    st.stop()

try:
    df = aplicar_qualidade(carregar_xlsx(arquivo))
except Exception as erro:  # arquivo fora do formato esperado
    st.error(f"Não consegui ler a planilha: {erro}")
    st.stop()

try:
    with Catalogo(CATALOGO_DB) as catalogo:
        previa = catalogo.importar(df, sincronizar=sincronizar, simular=True)
except ValueError as erro:
    st.error(f"Planilha recusada: {erro}")
    st.stop()

st.subheader("O que muda (pré-visualização, nada foi gravado)")
colunas = st.columns(5)
colunas[0].metric("Novos", previa.inseridos)
colunas[1].metric("Atualizados", previa.atualizados, help=f"{previa.precos_alterados} com preço alterado")
colunas[2].metric("Inalterados", previa.inalterados)
colunas[3].metric("Reativados", previa.reativados)
colunas[4].metric("Desativados", previa.desativados)
st.caption(
    f"{len(df)} linhas na planilha · {df['fornecedor'].nunique()} fornecedores · "
    f"{previa.textos_novos} descrições novas a vetorizar (≈ {previa.textos_novos / 10:.0f} s com o Ollama)."
)
if previa.desativados and not sincronizar:
    st.warning("Há itens a desativar, mas a opção acima está desmarcada: nenhum será desativado.")

if st.button("Aplicar atualização", type="primary"):
    with Catalogo(CATALOGO_DB) as catalogo:
        feito = catalogo.importar(df, sincronizar=sincronizar)
    carregar.clear()  # a busca recarrega o catalogo e vetoriza so as descricoes novas
    with st.spinner("Vetorizando as descrições novas…"):
        _, buscador, _ = carregar()
    st.success(f"Catálogo atualizado: {feito.resumo()}.")
    if buscador is not None and buscador.degradado:
        st.warning(
            "O Ollama está fora do ar: a busca usa só o modo lexical e as descrições novas ficam "
            "sem vetor até o servidor voltar (a busca refaz isso sozinha)."
        )
