"""Testes do resumo por regras (funcoes puras; sem LLM, sem Streamlit, sem rede)."""
import pandas as pd
import pytest

from autopecas import resumo
from autopecas.resumo import AVISO_REGRAS, escapar_markdown, resumir_itens, resumo_markdown


def itens(n=3, com_situacao=True):
    linhas = [
        ("N", "35027", "COXIM DO MOTOR - FUSCA / BRASILIA / KOMBI / VARIANTE /", 28.0, "cita kombi"),
        ("D", "160455", "COXIM DO MOTOR", 20.0, "sem veículo na descrição (genérico)"),
        ("L", "VW0585", "SUP CABO FREIO MAO LDO DIR", 8.0, "cita outro veículo: gol"),
        ("E", "7533349", "FILTRO OLEO", 31.0, "sem veículo na descrição (genérico)"),
    ][:n]
    df = pd.DataFrame([
        {"fornecedor": f, "codigo_interno": c, "descricao": d, "preco": p, "situacao_veiculo": s}
        for f, c, d, p, s in linhas
    ])
    return df if com_situacao else df.drop(columns="situacao_veiculo")


def test_uma_linha_por_item_na_ordem_da_lista_e_so_os_primeiros():
    linhas = resumir_itens(itens(4), max_itens=3)
    assert [ln.codigo for ln in linhas] == ["35027", "160455", "VW0585"]
    assert linhas[0].rotulo == "N · 35027" and linhas[0].preco == "R$ 28,00"
    assert linhas[0].veiculo == "cita kombi"


def test_descricao_vai_na_integra_sem_parafrase():
    ln = resumir_itens(itens(3))[2]
    assert ln.descricao == "SUP CABO FREIO MAO LDO DIR"  # o modelo leria isto como "cabo": o texto nao interpreta


def test_descricao_muito_longa_e_cortada_com_reticencias():
    df = itens(1)
    df.loc[0, "descricao"] = "X" * 500
    ln = resumir_itens(df)[0]
    assert len(ln.descricao) == resumo.LIMITE_DESCRICAO_RESUMO and ln.descricao.endswith("…")


def test_sem_coluna_de_situacao_ou_sem_itens_nao_quebra():
    assert resumir_itens(itens(2, com_situacao=False))[0].veiculo == ""
    assert resumir_itens(itens(0)) == []
    assert resumo_markdown([]) == ""


def test_preco_ausente_vira_nd_e_texto_de_controle_e_removido():
    df = itens(1)
    df.loc[0, "preco"] = None
    df.loc[0, "descricao"] = "COXIM\nDO\tMOTOR\x00"
    ln = resumir_itens(df)[0]
    assert ln.preco == "preço n/d" and ln.descricao == "COXIM DO MOTOR"


def test_markdown_escapa_cifrao_e_formatacao_do_catalogo():
    df = itens(1)
    df.loc[0, "descricao"] = "JOGO *PROMO* _2_ [kit] $x$"
    md = resumo_markdown(resumir_itens(df))
    assert "R\\$ 28,00" in md
    assert "\\*PROMO\\*" in md and "\\_2\\_" in md and "\\[kit\\]" in md and "\\$x\\$" in md
    assert md.startswith("- **N · 35027**")


def test_escapar_markdown_nao_altera_texto_comum():
    assert escapar_markdown("cabo de freio 2230 mm") == "cabo de freio 2230 mm"


def test_aviso_nao_promete_compatibilidade():
    base = AVISO_REGRAS.lower()
    assert "sem ia" in base and "sugestão a validar" in base and "não confirma" in base


def test_nenhum_valor_do_catalogo_aparece_como_afirmacao_de_peca():
    # o resumo so reproduz campos: nenhuma frase propria diz "e a peca pedida", "serve" ou "compativel"
    md = resumo_markdown(resumir_itens(itens(4))).lower()
    for proibida in ("é a peça", "serve", "compatível", "solicitad"):
        assert proibida not in md
