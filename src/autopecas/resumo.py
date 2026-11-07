"""Resumo por regras dos primeiros itens da lista (decisao 59). Motor: sem LLM, sem Streamlit, sem rede.

Por que existe: nas medicoes de 07/11/2026 (12 consultas, 2 modelos, 3 prompts) os modelos pequenos
(gemma3:1b e llama3.2:3b) nao identificaram a peca de forma confiavel (confundiram suporte com cabo,
coxim com retrovisor) e a guarda de texto nao pega esses erros. Aqui nada e parafraseado: cada linha traz
o fornecedor, o codigo, a descricao do catalogo na integra, o preco e a situacao do veiculo que a propria
busca calculou. Quem le confere a descricao; o resumo NAO diz que o item e a peca pedida nem que serve.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autopecas.llm import formatar_preco, limpar_texto

MAX_ITENS_RESUMO = 5
LIMITE_DESCRICAO_RESUMO = 160

AVISO_REGRAS = (
    "Resumo montado pelo código a partir dos campos do catálogo, sem IA: não confirma que o item é a peça "
    "pedida nem a compatibilidade com o veículo. Sugestão a validar: confira a descrição e a aplicação "
    "antes de vender."
)


@dataclass(frozen=True)
class LinhaResumo:
    fornecedor: str
    codigo: str
    descricao: str
    preco: str
    veiculo: str  # "" quando a consulta nao cita veiculo

    @property
    def rotulo(self) -> str:
        return f"{self.fornecedor} · {self.codigo}"


def _cortar(texto: str, limite: int) -> str:
    return texto if len(texto) <= limite else texto[: limite - 1].rstrip() + "…"


def resumir_itens(itens: pd.DataFrame, max_itens: int = MAX_ITENS_RESUMO) -> list[LinhaResumo]:
    """Uma linha por item, na ordem da lista (so os `max_itens` primeiros). Lista vazia se nao ha itens."""
    tem_situacao = "situacao_veiculo" in itens.columns
    linhas = []
    for _, item in itens.head(max_itens).iterrows():
        linhas.append(LinhaResumo(
            fornecedor=limpar_texto(item["fornecedor"], 8),
            codigo=limpar_texto(item["codigo_interno"], 40),
            descricao=_cortar(limpar_texto(item["descricao"], 10_000), LIMITE_DESCRICAO_RESUMO),
            preco=formatar_preco(item["preco"]),
            veiculo=limpar_texto(item["situacao_veiculo"], 120) if tem_situacao else "",
        ))
    return linhas


_ESPECIAIS = set("\\`*_{}[]<>#+!|~$")


def escapar_markdown(texto: str) -> str:
    """Texto do catalogo ou do modelo e dado, nao formatacao: '$' viraria LaTeX e '_'/'*' viraria estilo."""
    return "".join("\\" + c if c in _ESPECIAIS else c for c in texto)


def resumo_markdown(linhas: list[LinhaResumo]) -> str:
    """Lista em markdown, com tudo escapado. Vazio se nao ha linhas."""
    saida = []
    for ln in linhas:
        partes = [f"**{escapar_markdown(ln.rotulo)}**", escapar_markdown(ln.descricao), escapar_markdown(ln.preco)]
        if ln.veiculo:
            partes.append(escapar_markdown(ln.veiculo))
        saida.append("- " + " — ".join(partes))
    return "\n".join(saida)
