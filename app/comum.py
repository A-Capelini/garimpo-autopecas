"""Pedacos compartilhados pelas paginas do app (cache do buscador, mensagens)."""
from __future__ import annotations

import streamlit as st

from autopecas import llm
from autopecas.config import CATALOGO_DB
from autopecas.resumo import AVISO_REGRAS, escapar_markdown, resumir_itens, resumo_markdown
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


def _info_geracao(exp) -> str:
    partes = [exp.modelo]
    if exp.t_primeira_palavra is not None:
        partes.append(f"1ª palavra {exp.t_primeira_palavra:.1f} s")
    if exp.t_total is not None:
        partes.append(f"total {exp.t_total:.1f} s")
    if exp.estatisticas is not None and exp.estatisticas.eval_rate:
        partes.append(f"{exp.estatisticas.eval_rate:.1f} tok/s")
    return " · ".join(partes)


def _secao_llm(consulta: str, itens_lista, chave) -> None:
    """Resumo em linguagem natural: opcional, desligado por padrao. Qualquer falha deixa a lista e o resumo
    por regras como estao (o LLM nunca derruba a tela)."""
    modelo = llm.MODELO_PADRAO
    st.caption(llm.AVISO_LLM_EXPERIMENTAL)
    if not st.toggle(f"Ativar resumo por IA local ({modelo})", key="llm_ativo"):
        return
    if not llm.modelo_disponivel(modelo):
        st.warning(f"O modelo {modelo} não está instalado ou o Ollama está parado. Para instalar: ollama pull {modelo}")
        return
    if not st.session_state.get("_llm_aquecido"):
        with st.spinner("Carregando o modelo e preparando o prompt (pode levar até 30 s na primeira vez)…"):
            carregado = llm.aquecer_modelo(modelo)
        if not carregado:
            st.warning("O Ollama não respondeu ao carregar o modelo. A lista e o resumo por regras continuam valendo.")
            return
        st.session_state["_llm_aquecido"] = True

    if st.button("Gerar resumo com IA local", key="btn_llm"):
        exp = llm.explicar(consulta, itens_lista, modelo=modelo)
        if exp is None:
            st.info("Não há itens para resumir.")
            return
        caixa = st.empty()
        with caixa.container():
            st.write_stream(escapar_markdown(pedaco) for pedaco in exp)
        caixa.empty()  # o texto definitivo (ja verificado e sem frase cortada) e mostrado abaixo
        if exp.bloqueada:
            guardado = {"chave": chave, "erro": "A verificação automática bloqueou este resumo (afirmava "
                        "compatibilidade ou citava item fora da lista). Use o resumo por regras."}
        elif exp.erro:
            guardado = {"chave": chave, "erro": exp.erro + " A lista e o resumo por regras continuam disponíveis."}
        else:
            guardado = {"chave": chave, "texto": exp.texto_final, "info": _info_geracao(exp)}
        st.session_state["_llm_resumo"] = guardado

    guardado = st.session_state.get("_llm_resumo")
    if guardado and guardado["chave"] == chave:
        if guardado.get("erro"):
            st.warning(guardado["erro"])
        else:
            st.markdown(escapar_markdown(guardado["texto"]))
            if llm.indica_recusa(guardado["texto"]):
                st.info("A IA local disse que não encontrou a peça entre os primeiros itens. Isso pode estar errado "
                        "(modelos pequenos recusam demais): confira a lista e o resumo por regras acima.")
            st.caption(llm.AVISO_LLM)
            st.caption(guardado["info"])


def secao_resumo(consulta: str, itens_lista, chave) -> None:
    """Resumo dos primeiros itens: por regras (sempre) e por IA local (opcional). Aparece depois da lista."""
    st.subheader("Resumo dos primeiros itens")
    st.markdown(resumo_markdown(resumir_itens(itens_lista)))
    st.caption(AVISO_REGRAS)
    with st.expander("Resumo em linguagem natural (IA local, experimental)"):
        _secao_llm(consulta, itens_lista, chave)
