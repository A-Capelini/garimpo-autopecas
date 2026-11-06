import numpy as np
import pandas as pd
import pytest
from helpers import df_de, linha

from autopecas.evaluation.gabarito_texto import (
    ErroGabarito,
    avisos_do_gabarito,
    itens_do_nivel,
    ler_gabarito,
    resolver,
    resumo_do_gabarito,
    validar_contra_catalogo,
)

CABECALHO = "id;pergunta;pecas;tipo;autor;observacao\n"


def escrever(tmp_path, texto, nome="g.csv", codificacao="utf-8"):
    caminho = tmp_path / nome
    caminho.write_bytes(texto.encode(codificacao))
    return caminho


def catalogo():
    return df_de(
        linha("A", "07624", "cabo capo gol"),
        linha("A", "00010", "filtro oleo uno"),
        linha("B", "B-1", "cabo capo gol similar"),
        linha("C", "C1", "vela palio"),
        linha("D", "D1", "amortecedor fusca"),
        linha("E", "E1", "item de fora do basico"),
    )


# ---------------------------------------------------------------- leitura
def test_le_perguntas_com_pecas_tipo_e_autor(tmp_path):
    f = escrever(tmp_path, CABECALHO + "Q01;cabo do capô;A:07624|B:B-1;direta;Ana;obs\nQ02;vela;;sem_resposta;;\n")
    p = ler_gabarito(f)
    assert [x.id for x in p] == ["Q01", "Q02"]
    assert p[0].pecas == (("A", "07624"), ("B", "B-1"))  # zero a esquerda preservado (texto)
    assert (p[0].tipo, p[0].autor, p[0].linha) == ("direta", "Ana", 2)
    assert p[1].pecas == () and p[1].sem_resposta


def test_aceita_bom_virgula_e_cabecalho_com_acento(tmp_path):
    f = escrever(tmp_path, "﻿id,pergunta,peças\nQ01,cabo do capô,A:07624\n")
    p = ler_gabarito(f)
    assert p[0].pecas == (("A", "07624"),) and p[0].tipo == "nao_informado"


def test_aceita_windows_1252_do_excel(tmp_path):
    f = escrever(tmp_path, CABECALHO + "Q01;cabo do capô;A:07624;direta;José;\n", codificacao="cp1252")
    p = ler_gabarito(f)
    assert p[0].texto == "cabo do capô" and p[0].autor == "José"


def test_celula_entre_aspas_pode_usar_ponto_e_virgula_e_ignora_linhas_em_branco(tmp_path):
    f = escrever(tmp_path, CABECALHO + 'Q01;cabo;"A:07624;B:B-1";;;\n;;;;;\n\n')
    p = ler_gabarito(f)
    assert len(p) == 1 and p[0].pecas == (("A", "07624"), ("B", "B-1"))


def test_fornecedor_em_minuscula_vira_maiuscula_e_espacos_sao_aparados(tmp_path):
    f = escrever(tmp_path, CABECALHO + "Q01; cabo ; a : 07624 ;;;\n")
    p = ler_gabarito(f)
    assert p[0].pecas == (("A", "07624"),) and p[0].texto == "cabo"


def test_acumula_todos_os_erros_com_numero_da_linha(tmp_path):
    f = escrever(
        tmp_path,
        CABECALHO
        + "Q01;;A:1;direta;;\n"  # pergunta vazia
        + "Q01;vela;A:1;direta;;\n"  # id repetido
        + "Q03;vela;07624;direta;;\n"  # peca sem fornecedor
        + "Q04;vela;;direta;;\n"  # sem pecas e nao e sem_resposta
        + "Q05;vela;A:1;sem_resposta;;\n"  # sem_resposta com peca
        + "Q06;vela;A:1;inventado;;\n"  # tipo invalido
        + "Q07;vela;A:1|A:1;direta;;\n",  # peca repetida
    )
    with pytest.raises(ErroGabarito) as e:
        ler_gabarito(f)
    msg = str(e.value)
    assert len(e.value.problemas) == 7
    for trecho in ("linha 2", "pergunta vazia", "repetido", "fora do formato", "sem peças esperadas",
                   "sem_resposta não pode", "inválido", "repetida na mesma pergunta"):
        assert trecho in msg


def test_cabecalho_sem_coluna_obrigatoria_e_arquivo_vazio(tmp_path):
    with pytest.raises(ErroGabarito, match="Faltam colunas.*pecas"):
        ler_gabarito(escrever(tmp_path, "id;pergunta\nQ1;x\n"))
    with pytest.raises(ErroGabarito, match="vazio"):
        ler_gabarito(escrever(tmp_path, "\n\n", nome="v.csv"))


def test_modelo_do_repositorio_e_valido_no_formato():
    from pathlib import Path

    modelo = Path(__file__).resolve().parents[1] / "docs" / "exemplos" / "gabarito_basico_modelo.csv"
    p = ler_gabarito(modelo)
    assert [x.id for x in p] == ["Q01", "Q02", "Q03"]
    assert p[2].sem_resposta and len(p[1].pecas) == 2


# ---------------------------------------------------------------- catalogo
def test_validacao_aponta_fornecedor_fora_do_nivel_e_codigo_inexistente(tmp_path):
    f = escrever(tmp_path, CABECALHO + "Q01;a;E:E1;direta;;\nQ02;b;A:999;direta;;\nQ03;c;A:07624;direta;;\n")
    erros = validar_contra_catalogo(ler_gabarito(f), catalogo(), "Básico")
    assert len(erros) == 2
    assert "fornecedor E não faz parte do nível Básico" in erros[0]
    assert "código não existe" in erros[1] and "linha 3" in erros[1]


def test_validacao_sugere_codigo_quando_o_excel_removeu_zero_a_esquerda(tmp_path):
    f = escrever(tmp_path, CABECALHO + "Q01;a;A:7624;direta;;\n")
    erros = validar_contra_catalogo(ler_gabarito(f), catalogo(), "Básico")
    assert "Você quis dizer A:07624" in erros[0]


def test_resolver_converte_pecas_em_posicoes_do_nivel(tmp_path):
    f = escrever(tmp_path, CABECALHO + "Q01;cabo;A:07624|B:B-1;direta;;\nQ02;x;;sem_resposta;;\nQ03;vela;C:C1;;;\n")
    perguntas = ler_gabarito(f)
    gab = resolver(perguntas, catalogo(), "Básico")
    sub = itens_do_nivel(catalogo(), "Básico")
    assert len(sub) == 5 and "E" not in set(sub["fornecedor"])  # E fica fora do Basico
    assert gab.relevantes[0].tolist() == [0, 2]
    assert gab.relevantes[1].tolist() == []
    assert gab.relevantes[2].tolist() == [3]
    assert gab.com_resposta.tolist() == [True, False, True]
    assert gab.tipos == ("direta", "sem_resposta", "nao_informado")


def test_resolver_inclui_todas_as_linhas_quando_o_catalogo_repete_a_peca(tmp_path):
    df = df_de(linha("A", "1", "x"), linha("A", "1", "x repetido"), linha("B", "2", "y"))
    gab = resolver(ler_gabarito(escrever(tmp_path, CABECALHO + "Q1;x;A:1;direta;;\n")), df, "Básico")
    assert gab.relevantes[0].tolist() == [0, 1]


def test_resolver_recusa_gabarito_que_nao_bate_com_o_catalogo(tmp_path):
    perguntas = ler_gabarito(escrever(tmp_path, CABECALHO + "Q1;x;A:999;direta;;\n"))
    with pytest.raises(ErroGabarito, match="não existe"):
        resolver(perguntas, catalogo(), "Básico")


def test_avisos_e_resumo(tmp_path):
    linhas = "".join(f"Q{i};p{i};A:07624;direta;Ana;\n" for i in range(29)) + "S1;nada;;sem_resposta;Bia;\n"
    perguntas = ler_gabarito(escrever(tmp_path, CABECALHO + linhas))
    assert "29" in avisos_do_gabarito(perguntas)[0]  # 29 com resposta < 30
    r = resumo_do_gabarito(perguntas)
    assert (r["perguntas"], r["com_resposta"], r["sem_resposta"], r["pecas_distintas"]) == (30, 29, 1, 1)
    assert r["por_tipo"] == {"direta": 29, "sem_resposta": 1} and r["por_autor"] == {"Ana": 29, "Bia": 1}
    mais = perguntas + [perguntas[0].__class__("Q99", "p", (("A", "07624"),), "direta", "", "", 99)]
    assert avisos_do_gabarito(mais) == []  # 30 com resposta: criterio atendido
