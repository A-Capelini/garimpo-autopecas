"""Sugestao de pecas do holdout por filtro de palavras-chave. Descricoes sinteticas (decisao 27)."""
import pandas as pd
import pytest
from helpers import df_de, linha

from autopecas.evaluation.conferencia_holdout import escrever_conferencia, ler_conferencia
from autopecas.evaluation.gabarito_texto import itens_do_nivel
from autopecas.evaluation.sugestao_holdout import casa, gabarito_de_conferencia, montar_indice, sugerir


def indice():
    df = df_de(
        linha("A", "001", "BOMBA D AGUA MONZA 83/96"),
        linha("A", "002", "BOMBA D AGUA GOL 85/94"),
        linha("B", "B01", "BOMBA DA AGUA GENERICA"),
        linha("A", "003", "AMORTECEDOR DIANT. GOL 85/94"),
        linha("A", "004", "AMORTECEDOR TRAS. GOL 85/94"),
        linha("A", "005", "FAROL PALIO 96/00 LD"),
        linha("A", "006", "FAROL PALIO 96/00 LE"),
        linha("C", "00840", "INTERR. OLEO OPALA 69/92"),
        linha("C", "007", "COIFA HOMOCINETICA GOL"),
        linha("D", "D01", "TAMPA DO RADIADOR"),
        linha("E", "E01", "BOMBA DAGUA MONZA FORA DO BASICO"),
    )
    return montar_indice(itens_do_nivel(df, "Básico"), "Básico")


def codigos(s):
    return [(c.fornecedor, c.codigo) for c in s.candidatas]


# ---------------------------------------------------------------- casamento de termos
@pytest.mark.parametrize("token, palavra, esperado", [
    ("bomba", "bomba", "exato"),
    ("velas", "vela", "exato"),               # plural
    ("interruptor", "interr", "abreviacao"),   # o catalogo abrevia
    ("diant", "dianteiro", "abreviacao"),      # a pergunta abrevia
    ("tampao", "tampa", None),                 # tampao != tampa (diferenca de 1 letra nao e abreviacao)
    ("12v", "12v", "exato"),
    ("12v", "12", None),                       # numeros so por igualdade
    ("freio", "freios", "exato"),
    ("carro", "arro", None),
])
def test_casa(token, palavra, esperado):
    assert casa(token, palavra) == esperado


# ---------------------------------------------------------------- filtro e triagem
def test_com_veiculo_so_o_que_cita_o_veiculo_e_provavel_e_generico_e_improvavel():
    s = sugerir("H01", "bomba de água do monza", indice())
    por_cod = {(c.fornecedor, c.codigo): c for c in s.candidatas}
    assert por_cod[("A", "001")].triagem == "provavel"
    assert por_cod[("A", "002")].triagem == "improvavel" and "outro veículo" in por_cod[("A", "002")].motivo
    assert por_cod[("B", "B01")].triagem == "improvavel" and "genérica" in por_cod[("B", "B01")].motivo
    assert ("E", "E01") not in por_cod  # fora do nivel
    assert s.existe_peca == "sim" and s.veiculos == ("monza",)
    assert codigos(s)[0] == ("A", "001")  # provavel primeiro


def test_sem_veiculo_entram_todas_do_nivel_como_provaveis():
    s = sugerir("H34", "bomba de água", indice())
    assert {c.triagem for c in s.candidatas} == {"provavel"}
    assert set(codigos(s)) == {("A", "001"), ("A", "002"), ("B", "B01")}


def test_posicao_conflitante_vira_improvavel_e_sem_lado_serve_ld_e_le():
    s = sugerir("H65", "amortecedor dianteiro do gol", indice())
    por_cod = {c.codigo: c for c in s.candidatas}
    assert por_cod["003"].triagem == "provavel"
    assert por_cod["004"].triagem == "improvavel" and "posição" in por_cod["004"].conflitos
    s2 = sugerir("Q", "farol do palio", indice())  # sem lado: os dois servem
    assert {c.codigo: c.triagem for c in s2.candidatas} == {"005": "provavel", "006": "provavel"}


def test_lado_pedido_descarta_o_lado_oposto_e_a_palavra_lado_nao_e_exigida():
    s = sugerir("Q12", "farol do palio lado direito", indice())
    por_cod = {c.codigo: c.triagem for c in s.candidatas}
    assert por_cod == {"005": "provavel", "006": "improvavel"}
    assert "lado" not in s.termos


def test_abreviacao_do_catalogo_e_erro_de_digitacao():
    s = sugerir("H07", "interruptor de óleo do opala", indice())
    assert codigos(s) == [("C", "00840")] and s.candidatas[0].triagem == "provavel"
    s2 = sugerir("H75", "tampa do radiadro", indice())  # erro de digitacao: casa por similaridade de grafia
    assert codigos(s2) == [("D", "D01")]
    assert "~" in s2.candidatas[0].casamentos
    assert s2.candidatas[0].triagem == "possivel"  # grafia aproximada nunca e "provavel"


def test_termo_sem_correspondencia_e_ignorado_no_filtro_mas_rebaixa_para_possivel():
    s = sugerir("H13", "silencioso do gol", indice())  # 'silencioso' nao existe; 'gol' e veiculo
    assert s.sem_correspondencia == ("silencioso",) and s.termos == ()
    assert s.candidatas == () and s.existe_peca == "nada"
    s2 = sugerir("Hx", "amortecedor xyzzyqq do gol", indice())
    assert s2.sem_correspondencia == ("xyzzyqq",)
    assert {c.triagem for c in s2.candidatas if c.codigo in ("003", "004")} == {"possivel"}


def test_so_improvavel_quando_so_ha_item_generico_ou_de_outro_veiculo():
    s = sugerir("H27", "tampa do radiador do monza", indice())
    assert s.existe_peca == "so_improvavel" and codigos(s) == [("D", "D01")]


def test_peca_parecida_aparece_para_o_grupo_decidir():
    # a triagem NAO decide "coifa != junta homocinetica": lista e o grupo marca N
    s = sugerir("Q", "coifa do gol", indice())
    assert ("C", "007") in codigos(s)


def test_camada_parcial_so_com_tres_termos_ou_mais():
    df = df_de(linha("A", "1", "CABO FREIO MAO BRASILIA"), linha("A", "2", "CABO FREIO GOL"))
    ind = montar_indice(itens_do_nivel(df, "Básico"))
    s = sugerir("Q", "cabo do freio de mão da brasília", ind)
    por_cod = {c.codigo: c for c in s.candidatas}
    assert por_cod["1"].camada == "completa" and por_cod["1"].triagem == "provavel"
    assert por_cod["2"].camada == "parcial" and por_cod["2"].triagem == "improvavel"  # outro veiculo/generico
    s2 = sugerir("Q", "cabo do freio", ind)  # 2 termos: sem camada parcial
    assert all(c.camada == "completa" for c in s2.candidatas)


def test_corte_de_candidatas_preserva_o_total():
    df = df_de(*[linha("A", f"{i:03d}", f"PARAFUSO TIPO {i}") for i in range(50)])
    ind = montar_indice(itens_do_nivel(df, "Básico"))
    s = sugerir("Q", "parafuso", ind, max_candidatas=10)
    assert len(s.candidatas) == 10 and s.total_filtro == 50 and s.contagem == {"provavel": 50}


def test_ordem_deterministica():
    a = sugerir("Q", "bomba de água do monza", indice())
    b = sugerir("Q", "bomba de água do monza", indice())
    assert codigos(a) == codigos(b)


# ---------------------------------------------------------------- fechamento
def perguntas():
    return pd.DataFrame([
        {"id": "H01", "pergunta": "bomba de água do monza", "tipo": "direta", "autor": "Gemini", "observacao": ""},
        {"id": "H02", "pergunta": "virabrequim da brasília", "tipo": "sem_resposta", "autor": "Gemini", "observacao": "x"},
        {"id": "H03", "pergunta": "farol do palio", "tipo": "direta", "autor": "Gemini", "observacao": ""},
        {"id": "H04", "pergunta": "vela", "tipo": "sem_resposta", "autor": "Gemini", "observacao": ""},
    ])


def revisadas(*itens):
    return pd.DataFrame(itens, columns=["id", "fornecedor", "codigo", "serve", "revisor"])


def test_fechamento_monta_pecas_tipo_e_revisores():
    rev = revisadas(("H01", "A", "001", "S", "Ana"), ("H01", "A", "002", "N", "Ana"),
                    ("H03", "A", "005", "S", "Bia"), ("H03", "A", "006", "S", "Bia"))
    gab, pend = gabarito_de_conferencia(perguntas(), rev, {"H04": "A:900"})
    assert pend == []
    por_id = gab.set_index("id")
    assert por_id.loc["H01", "pecas"] == "A:001" and por_id.loc["H01", "tipo"] == "direta"
    assert "revisado por Ana" in por_id.loc["H01", "observacao"]
    assert por_id.loc["H03", "pecas"] == "A:005|A:006"
    assert por_id.loc["H02", "pecas"] == "" and por_id.loc["H02", "tipo"] == "sem_resposta"
    # sem_resposta do Gemini que ganhou peca (extra do grupo): fica sem tipo, com aviso na observacao
    assert por_id.loc["H04", "pecas"] == "A:900" and por_id.loc["H04", "tipo"] == ""
    assert "achou peça" in por_id.loc["H04", "observacao"]
    assert list(gab.columns) == ["id", "pergunta", "pecas", "tipo", "autor", "observacao"]


def test_pergunta_sem_nenhum_s_vira_sem_resposta_e_registra_o_tipo_original():
    rev = revisadas(("H01", "A", "001", "N", "Ana"))
    gab, pend = gabarito_de_conferencia(perguntas(), rev)
    linha_h01 = gab.set_index("id").loc["H01"]
    assert pend == [] and linha_h01["tipo"] == "sem_resposta" and "tipo original direta" in linha_h01["observacao"]


@pytest.mark.parametrize("serve", ["", "?", "talvez", None])
def test_fechamento_recusa_decisao_pendente(serve):
    gab, pend = gabarito_de_conferencia(perguntas(), revisadas(("H01", "A", "001", serve, "Ana")))
    assert len(pend) == 1 and "sem decisão" in pend[0]


def test_fechamento_exige_revisor_e_id_valido_e_formato_do_extra():
    rev = revisadas(("H01", "A", "001", "S", ""), ("H99", "A", "002", "S", "Ana"))
    _, pend = gabarito_de_conferencia(perguntas(), rev, {"H03": "sem-formato"})
    assert any("sem o nome" in p for p in pend)
    assert any("H99" in p and "não existe" in p for p in pend)
    assert any("fora do formato" in p for p in pend)


def test_fechamento_por_nivel_ignora_fornecedores_de_fora_inclusive_pendencias():
    rev = revisadas(("H01", "A", "001", "S", "Ana"), ("H01", "F", "F9", "S", "Ana"),
                    ("H03", "G", "G1", ""  , ""))  # G fora do Basico: sem decisao nao e pendencia nele
    basico, pend = gabarito_de_conferencia(perguntas(), rev, {"H03": "F:7"}, frozenset("ABCD"))
    por_id = basico.set_index("id")
    assert pend == [] and por_id.loc["H01", "pecas"] == "A:001"
    assert por_id.loc["H03", "pecas"] == "" and por_id.loc["H03", "tipo"] == "sem_resposta"
    inter, pend = gabarito_de_conferencia(perguntas(), rev, {"H03": "F:7"}, frozenset("ABCDEFGHI"))
    assert any("G:G1" in p for p in pend)  # no Intermediario a decisao que falta aparece
    assert inter.set_index("id").loc["H01", "pecas"] == "A:001|F:F9"


def test_aceita_s_minusculo():
    gab, pend = gabarito_de_conferencia(perguntas(), revisadas(("H01", "A", "001", "s", "Ana")))
    assert pend == [] and gab.set_index("id").loc["H01", "pecas"] == "A:001"


# ---------------------------------------------------------------- planilha (ida e volta)
def test_planilha_ida_e_volta_preserva_zeros_e_le_decisoes(tmp_path):
    from openpyxl import load_workbook

    sugs = [sugerir("H07", "interruptor de óleo do opala", indice()), sugerir("H01", "bomba de água do monza", indice())]
    caminho = tmp_path / "c.xlsx"
    escrever_conferencia(caminho, sugs, {"H07": "direta", "H01": "direta"}, "Básico", "abc123")
    wb = load_workbook(caminho)
    assert wb.sheetnames == ["Instruções", "Perguntas", "Candidatas", "Descartadas"]
    assert [r[2].value for r in wb["Descartadas"].iter_rows(min_row=2)] == ["A", "B"]  # improvaveis de H01 saem da conferencia
    assert wb["Candidatas"].max_row == 3  # so os 2 provaveis (H07 e H01)
    ws = wb["Candidatas"]
    assert ws["D2"].value == "00840" and ws["D2"].number_format == "@"  # zero a esquerda como texto
    ws["I2"], ws["J2"] = "S", "Ana"
    ws["I3"], ws["J3"] = "N", "Ana"
    wb["Perguntas"]["M2"] = "A:001"
    wb.save(caminho)

    rev, extras = ler_conferencia(caminho)
    assert list(rev.columns) == ["id", "fornecedor", "codigo", "serve", "revisor"]
    assert rev.loc[0, "codigo"] == "00840" and rev.loc[0, "serve"] == "S" and len(rev) == 2
    assert extras == {"H07": "A:001"}
    gab, pend = gabarito_de_conferencia(
        pd.DataFrame([{"id": "H07", "pergunta": "x", "tipo": "direta", "autor": "Gemini", "observacao": ""},
                      {"id": "H01", "pergunta": "y", "tipo": "direta", "autor": "Gemini", "observacao": ""}]),
        rev, extras)
    assert gab.set_index("id").loc["H07", "pecas"].startswith("C:00840")  # zero preservado ate o gabarito


def test_leitura_recusa_planilha_sem_aba(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    wb.save(tmp_path / "x.xlsx")
    with pytest.raises(ValueError, match="Falta a aba"):
        ler_conferencia(tmp_path / "x.xlsx")
