"""Modo Adaptações: sugestões por veículo, família no catálogo e faixa de ano. Sempre "a validar".

So interface: toda a logica esta em src/autopecas/adaptacoes.py (motor, sem Streamlit). Nao usa LLM.
"""
from __future__ import annotations

import streamlit as st

from autopecas.adaptacoes import AVISO, ROTULO_DECLARADO, sugerir
from autopecas.veiculos import VEICULOS, analisar_consulta
from comum import atualizar_status, aviso_catalogo_vazio, carregar

st.set_page_config(page_title="Garimpo · Adaptações", page_icon="🔧", layout="wide")

itens, buscador, _ = carregar()
st.title("🔧 Adaptações")
if buscador is None:
    aviso_catalogo_vazio()
    st.stop()
atualizar_status(buscador)

st.info(AVISO)

lida = analisar_consulta(st.session_state.get("_ultima_consulta", ""))
opcoes = [""] + sorted(VEICULOS)
c1, c2, c3 = st.columns([3, 2, 1])
peca = c1.text_input("Peça", value=lida.peca, placeholder="ex.: cabo do capô")
veiculo = c2.selectbox(
    "Veículo", opcoes,
    index=opcoes.index(lida.veiculos[0]) if lida.veiculos and lida.veiculos[0] in opcoes else 0,
    format_func=lambda v: v.title() if v else "(escolha o veículo)",
)
ano_txt = c3.text_input("Ano (opcional)", value=str(lida.ano) if lida.ano else "", placeholder="ex.: 1988")

ano = None
if ano_txt.strip():
    if ano_txt.strip().isdigit() and 1950 <= int(ano_txt) <= 2035:
        ano = int(ano_txt)
    else:
        st.error("Ano inválido: use 4 dígitos entre 1950 e 2035 ou deixe em branco.")
        st.stop()

if not peca.strip() or not veiculo:
    st.info("Informe a peça e escolha o veículo para ver as adaptações.")
    st.stop()

resultado = sugerir(buscador, peca, veiculo, ano)
if resultado.total() == 0:
    st.warning("Nenhuma peça parecida encontrada para esse veículo e ano.")
    if resultado.fora_do_ano:
        st.caption(f"{resultado.fora_do_ano} item(ns) citam o veículo, mas com faixa de ano que não cobre {ano}.")
    st.stop()

for bloco in resultado.blocos:
    st.subheader(f"{bloco.titulo} ({len(bloco.itens)})")
    st.caption("Citado pelo fornecedor, não garantido." if bloco.rotulo == ROTULO_DECLARADO else "Sugestão a validar.")
    st.dataframe(
        bloco.itens.rename(columns={
            "fornecedor": "Fornecedor", "codigo_interno": "Código", "descricao": "Descrição", "preco": "Preço (R$)",
            "ncm": "NCM", "gtin": "GTIN", "rotulo": "Situação", "motivo": "Motivo",
        }),
        hide_index=True,
        column_config={"Preço (R$)": st.column_config.NumberColumn(format="%.2f")},
    )
if resultado.fora_do_ano:
    st.caption(f"{resultado.fora_do_ano} item(ns) citam o veículo ou a família, mas a faixa de ano não cobre {ano}: ocultos.")
st.caption("Famílias vêm do próprio catálogo (modelos citados juntos nas descrições), não de uma base externa de compatibilidade.")
