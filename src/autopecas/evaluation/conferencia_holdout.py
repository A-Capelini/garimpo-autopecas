"""Planilha de conferencia do holdout (gerar para o grupo preencher; ler de volta para fechar o gabarito).

A planilha contem descricoes do catalogo sigiloso: fica em `resultados/` (fora do Git) e nao vai a servico
externo. Abas:
- Instruções: criterio de rotulagem e como preencher.
- Perguntas: uma linha por pergunta (resumo da triagem) + coluna `pecas_extras` para o que o filtro nao achou.
- Candidatas: itens `provavel` e `possivel` do filtro, com `serve` (S/N) e `revisor` para o grupo preencher.
- Descartadas: itens que o filtro achou mas o criterio descarta (generico ou de outro veiculo quando a pergunta
  cita veiculo; lado/posicao contrario). Nao precisam de decisao; se o grupo discordar de alguma, acrescenta o
  codigo em `pecas_extras` na aba Perguntas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from autopecas.evaluation.sugestao_holdout import MAX_CANDIDATAS, Sugestao

FONTE = "Arial"
_CAB = PatternFill("solid", fgColor="1F3864")
_PREENCHER = PatternFill("solid", fgColor="FFFF00")

INSTRUCOES = [
    "Como preencher (cada integrante confere as suas perguntas; outra pessoa confere as dúvidas '?')",
    "1. Aba Candidatas: em cada linha, escreva S se a peça atende a pergunta, N se não atende. Escreva também o seu nome em `revisor`.",
    "2. '?' ou célula vazia não vale: a leitura final recusa o arquivo e lista o que falta. Resolva a dúvida com o grupo.",
    "3. Aba Perguntas: se existir peça que o filtro NÃO listou (ou que ele pôs em Descartadas e vocês acham que serve), escreva em `pecas_extras` no formato FORNECEDOR:código (ex.: C:02864), separadas por |.",
    "4. Não altere id, pergunta, fornecedor nem código. Não reescreva a pergunta para casar com uma peça (isso invalida o holdout).",
    "",
    "Critério de rotulagem (o mesmo do gabarito do Básico, docs/GABARITO_BASICO.md):",
    "- pergunta COM veículo: só entram peças cuja descrição cita aquele veículo; item genérico, sem veículo, NÃO entra;",
    "- pergunta SEM veículo: entram todos os itens daquela peça, de qualquer fornecedor do nível;",
    "- sem lado na pergunta, LD e LE servem; com lado, só o pedido;",
    "- peça parecida mas diferente não entra (coifa ≠ junta homocinética; tampa ≠ reservatório; bucha da bieleta ≠ bieleta);",
    "- compatibilidade não confirmada é 'sugestão a validar', nunca 'serve'.",
    "",
    "A coluna `triagem` (provavel / possivel / improvavel) é só uma dica do filtro de palavras-chave: a decisão é sua.",
    "Pergunta sem nenhum S vira `sem_resposta` (a peça não existe no nível). Isso conta como achado, não como erro.",
]


def _cabecalho(ws, linha, colunas):
    for j, nome in enumerate(colunas, 1):
        c = ws.cell(linha, j, nome)
        c.font = Font(name=FONTE, bold=True, color="FFFFFF")
        c.fill = _CAB
        c.alignment = Alignment(wrap_text=True, vertical="center")


def _estilizar(ws, primeira, ultima, ncols):
    for linha in ws.iter_rows(min_row=primeira, max_row=ultima, max_col=ncols):
        for c in linha:
            c.font = Font(name=FONTE, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")


def escrever_conferencia(
    caminho: Path, sugestoes: list[Sugestao], tipos: dict[str, str], nivel: str, sha_holdout: str
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Instruções"
    ws["A1"] = f"Conferência do holdout — nível {nivel} (holdout sha256 {sha_holdout}…)"
    ws["A1"].font = Font(name=FONTE, bold=True, size=13)
    for i, texto in enumerate(INSTRUCOES, 3):
        ws.cell(i, 1, texto).font = Font(name=FONTE, size=10, bold=(i == 3 or texto.startswith("Critério")))
    ws.cell(len(INSTRUCOES) + 4, 1, "Células amarelas = você preenche. Esta planilha tem descrições do catálogo: não suba a serviço externo nem ao Git.").font = Font(name=FONTE, size=9, italic=True)
    ws.column_dimensions["A"].width = 150

    wp = wb.create_sheet("Perguntas")
    colp = ["id", "pergunta", "tipo (Gemini)", "veículo", "ano", "termos da peça", "sem correspondência no catálogo",
            "itens no filtro", "provável", "possível", "improvável", "existe peça? (triagem)",
            "pecas_extras (FORNECEDOR:código|...)"]
    _cabecalho(wp, 1, colp)
    for i, s in enumerate(sugestoes, 2):
        valores = [s.id, s.pergunta, tipos.get(s.id, ""), ", ".join(s.veiculos), s.ano or "", " ".join(s.termos),
                   " ".join(s.sem_correspondencia), s.total_filtro, s.contagem.get("provavel", 0),
                   s.contagem.get("possivel", 0), s.contagem.get("improvavel", 0), s.existe_peca, ""]
        for j, v in enumerate(valores, 1):
            wp.cell(i, j, v)
        wp.cell(i, 13).fill = _PREENCHER
        wp.cell(i, 13).number_format = "@"
    _estilizar(wp, 2, len(sugestoes) + 1, len(colp))
    for j, w in enumerate([7, 46, 14, 14, 7, 28, 24, 10, 9, 9, 10, 14, 34], 1):
        wp.column_dimensions[get_column_letter(j)].width = w
    wp.freeze_panes = "C2"

    wc = wb.create_sheet("Candidatas")
    colc = ["id", "pergunta", "fornecedor", "codigo", "descrição", "triagem", "motivo da triagem", "termos casados",
            "serve (S/N)", "revisor", "observação"]
    _cabecalho(wc, 1, colc)
    linha = 2
    descartadas = []
    for s in sugestoes:
        for c in s.candidatas:
            if c.triagem == "improvavel":
                descartadas.append((s, c))
                continue
            for j, v in enumerate([s.id, s.pergunta, c.fornecedor, c.codigo, c.descricao, c.triagem, c.motivo,
                                   c.casamentos, "", "", ""], 1):
                wc.cell(linha, j, v)
            wc.cell(linha, 4).number_format = "@"  # texto: preserva zeros a esquerda
            for j in (9, 10, 11):
                wc.cell(linha, j).fill = _PREENCHER
            linha += 1
    _estilizar(wc, 2, max(linha - 1, 2), len(colc))
    for j, w in enumerate([7, 40, 10, 12, 60, 11, 38, 30, 11, 14, 28], 1):
        wc.column_dimensions[get_column_letter(j)].width = w
    wc.freeze_panes = "C2"
    if linha > 2:
        dv = DataValidation(type="list", formula1='"S,N"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Valor inválido", error="Use S ou N.")
        wc.add_data_validation(dv)
        dv.add(f"I2:I{linha - 1}")
        wc.auto_filter.ref = f"A1:{get_column_letter(len(colc))}{linha - 1}"
    wd = wb.create_sheet("Descartadas")
    cold = ["id", "pergunta", "fornecedor", "codigo", "descrição", "por que o critério descarta"]
    _cabecalho(wd, 1, cold)
    for i, (s, c) in enumerate(descartadas, 2):
        for j, v in enumerate([s.id, s.pergunta, c.fornecedor, c.codigo, c.descricao, c.motivo], 1):
            wd.cell(i, j, v)
        wd.cell(i, 4).number_format = "@"
    _estilizar(wd, 2, len(descartadas) + 1, len(cold))
    for j, w in enumerate([7, 40, 10, 12, 60, 60], 1):
        wd.column_dimensions[get_column_letter(j)].width = w
    wd.freeze_panes = "C2"
    wb.save(caminho)


def ler_conferencia(caminho: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Le a planilha preenchida: (candidatas com id/fornecedor/codigo/serve/revisor, extras por id)."""
    wb = load_workbook(caminho, data_only=True)
    for aba in ("Perguntas", "Candidatas"):
        if aba not in wb.sheetnames:
            raise ValueError(f"Falta a aba '{aba}' na planilha.")

    def tabela(nome):
        linhas = list(wb[nome].iter_rows(values_only=True))
        cab = [str(c).strip() if c is not None else "" for c in linhas[0]]
        return cab, [dict(zip(cab, l)) for l in linhas[1:] if any(v not in (None, "") for v in l)]

    _, cands = tabela("Candidatas")
    revisadas = pd.DataFrame([{
        "id": str(r["id"]).strip(), "fornecedor": str(r["fornecedor"]).strip(), "codigo": str(r["codigo"]).strip(),
        "serve": r.get("serve (S/N)") or "", "revisor": r.get("revisor") or "",
    } for r in cands], columns=["id", "fornecedor", "codigo", "serve", "revisor"])
    cab, pergs = tabela("Perguntas")
    chave = next((c for c in cab if c.startswith("pecas_extras")), None)
    extras = {str(r["id"]).strip(): str(r[chave]).strip() for r in pergs if chave and r.get(chave)}
    return revisadas, extras


__all__ = ["escrever_conferencia", "ler_conferencia", "MAX_CANDIDATAS"]
