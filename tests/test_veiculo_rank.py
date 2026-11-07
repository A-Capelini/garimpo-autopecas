"""Ordenacao por veiculo (retrieval/veiculo.py) e sua integracao no Buscador. Descricoes sinteticas."""
import numpy as np
import pandas as pd
import pytest

from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.retrieval.veiculo import (
    CAMADA_CITA, CAMADA_NEUTRA, CAMADA_OUTRO, ComVeiculo, casa, cobertura, palavras_da_peca, reordenar,
    situacao_do_item,
)
from autopecas.search import Buscador
from autopecas.veiculos import aplicar_extracao, analisar_consulta
from helpers import linha


def itens():
    d = [
        ("A", "1", "FILTRO OLEO MOTOR"),                   # generico, escrito por extenso
        ("B", "2", "MANGUEIRA FILTRO AR GOL"),             # cita gol, mas e outra peca
        ("C", "3", "FILTRO OLEO GOL"),                     # o certo
        ("D", "4", "FILTRO OLEO UNO"),                     # outro veiculo
        ("A", "5", "FILTRO OLEO GOL 85/90"),               # o certo, com faixa
        ("B", "6", "FILTRO OLEO GOL 95/99"),               # gol, ano fora
    ]
    from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA, normalizar_texto
    df = pd.DataFrame([linha(f, c, desc, texto=normalizar_texto(desc, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA))
                       for f, c, desc in d])
    return aplicar_extracao(df)


def ordem_base(df, consulta):
    r = TfidfRetriever()
    r.fit(df["texto_busca"].tolist())
    s = np.round(r.score_texts([consulta])[0], 10)
    o = np.argsort(-s, kind="stable")
    return o[s[o] > 0]


def test_casa_aceita_abreviacao_por_prefixo():
    assert casa("pastilha", "past")           # prefixo de 4 = 100% da menor palavra
    assert casa("acelerador", "acel")
    assert not casa("cabo", "cb")             # 2 letras: limite conhecido
    assert not casa("filtro", "freio")


def test_cobertura_e_palavras_da_peca():
    assert palavras_da_peca("filtro de oleo do motor") == ["filtro", "oleo", "motor"]
    assert cobertura(["filtro", "oleo"], "filtro oleo gol") == 1.0
    assert cobertura(["filtro", "oleo"], "mangueira filtro ar") == 0.5
    assert cobertura([], "qualquer") == 1.0


def test_situacao_do_item_camadas_e_textos():
    c = analisar_consulta("filtro oleo gol 1988")
    assert situacao_do_item((), None, None, c)[0] == CAMADA_NEUTRA
    assert situacao_do_item(("uno",), None, None, c)[0] == CAMADA_OUTRO
    assert situacao_do_item(("gol",), 1985, 1990, c)[0] == CAMADA_CITA
    cam, txt = situacao_do_item(("gol",), 1995, 1999, c)
    assert cam == CAMADA_OUTRO and "fora da faixa" in txt
    assert situacao_do_item(("gol",), None, None, c)[0] == CAMADA_CITA  # item sem ano nao e excluido
    assert situacao_do_item(("uno",), None, None, analisar_consulta("filtro oleo")) == (CAMADA_CITA, "")


def test_filtro_oleo_gol_coloca_o_especifico_antes_do_generico():
    df = itens()
    o = ordem_base(df, "filtro oleo gol")
    novo = reordenar(o, df, analisar_consulta("filtro oleo gol"))
    cod = lambda ordem: df["codigo_interno"].iloc[ordem].tolist()
    assert cod(novo)[0] in ("3", "5", "6")
    # a mangueira (gol, outra peca) NAO sobe: fica na camada neutra, junto do generico
    assert cod(novo).index("2") > cod(novo).index("3")
    # o outro veiculo vai para o fim
    assert cod(novo)[-1] == "4" or cod(novo).index("4") > cod(novo).index("1")
    assert sorted(novo.tolist()) == sorted(o.tolist())  # permutacao: nada some nem aparece


def test_ano_fora_da_faixa_vai_para_camada_3():
    df = itens()
    c = analisar_consulta("filtro oleo gol 1988")
    novo = reordenar(ordem_base(df, c.original), df, c)
    cod = df["codigo_interno"].iloc[novo].tolist()
    assert set(cod[:2]) == {"3", "5"} and cod.index("6") > cod.index("1")  # 95/99 nao cobre 1988; "3" nao tem ano


def test_consulta_sem_veiculo_nao_muda_nada():
    df = itens()
    o = ordem_base(df, "filtro oleo")
    assert reordenar(o, df, analisar_consulta("filtro oleo")).tolist() == o.tolist()


def test_so_reordena_o_topo():
    df = itens()
    o = ordem_base(df, "filtro oleo gol")
    novo = reordenar(o, df, analisar_consulta("filtro oleo gol"), profundidade=2)
    assert novo[2:].tolist() == o[2:].tolist()


def test_com_veiculo_zero_continua_zero_e_fit_exige_itens():
    df = itens()
    r = ComVeiculo(TfidfRetriever())
    assert r.nome == "tfidf-char+veiculo" or r.nome.endswith("+veiculo")
    with pytest.raises(TypeError):
        r.fit(df["texto_busca"].tolist())
    with pytest.raises(RuntimeError):
        r.score_texts(["filtro oleo gol"])
    r.fit_itens(df)
    S = r.score_texts(["filtro oleo gol", "zzzzz"])
    assert (S[1] == 0).all()                      # sem evidencia = zero
    assert S[0].max() == (S[0] > 0).sum()         # posicoes invertidas: 1..n
    melhor = int(np.argmax(S[0]))
    assert df["codigo_interno"].iloc[melhor] in ("3", "5", "6")


def test_com_veiculo_e_buscador_dao_a_mesma_ordem():
    df = itens()
    hib = HybridRetriever(TfidfRetriever(), _SemNada(), k_rrf=1, profundidade=50, fallback_lexical=True)
    b = Buscador(df, hib)
    # mesma base do app (o hibrido), para comparar so a regra; a base lexical pura pode desempatar diferente
    r = ComVeiculo(HybridRetriever(TfidfRetriever(), _SemNada(), k_rrf=1, profundidade=50, fallback_lexical=True))
    r.fit_itens(df)
    for consulta in ("filtro oleo gol", "filtro oleo gol 1988", "filtro oleo uno", "filtro oleo"):
        pelo_app = b.buscar(consulta, k=10).itens["codigo_interno"].tolist()
        S = r.score_texts([consulta])[0]
        o = np.argsort(-S, kind="stable")
        o = o[S[o] > 0]
        assert df["codigo_interno"].iloc[o].tolist()[:len(pelo_app)] == pelo_app


def test_buscador_expoe_camada_e_situacao():
    df = itens()
    hib = HybridRetriever(TfidfRetriever(), _SemNada(), k_rrf=1, profundidade=50, fallback_lexical=True)
    b = Buscador(df, hib)
    r = b.buscar("filtro oleo gol")
    assert {"camada", "situacao_veiculo"} <= set(r.itens.columns)
    assert r.consulta.veiculos == ("gol",)
    assert r.itens["situacao_veiculo"].str.len().gt(0).all()
    sem = b.buscar("filtro oleo")
    assert (sem.itens["situacao_veiculo"] == "").all()
    # por_veiculo=False desliga a regra (usado pelas Adaptacoes)
    base = b.buscar("filtro oleo gol", k=10, por_veiculo=False)
    assert len(base) > 0


class _SemNada:
    """Semantico que nao contribui (so para montar o hibrido)."""
    corpus_dependente = False
    nome = "nada"

    def fit(self, textos):
        self.n = len(textos)

    def score_rows(self, linhas):
        raise NotImplementedError

    def score_texts(self, consultas):
        return np.zeros((len(consultas), self.n))


def test_runner_mede_o_que_o_app_devolve_com_a_regra_do_veiculo():
    """avaliar_perguntas usa fit_itens (campos de veiculo/ano) e bate com a ordem do Buscador."""
    from autopecas.evaluation.gabarito_texto import Pergunta
    from autopecas.evaluation.runner_texto import avaliar_perguntas

    df = itens().assign(texto_busca=lambda d: d["texto_busca"])
    ps = [Pergunta("Q1", "filtro oleo gol", (("C", "3"),), "direta", "t", "", 2),
          Pergunta("Q2", "filtro oleo gol 1988", (("A", "5"),), "direta", "t", "", 3)]
    fabrica = lambda: HybridRetriever(TfidfRetriever(), _SemNada(), k_rrf=1, profundidade=50, fallback_lexical=True)
    base = avaliar_perguntas(df, ps, fabrica, nivel="Avançado")
    com = avaliar_perguntas(df, ps, lambda: ComVeiculo(fabrica()), nivel="Avançado")
    assert com.linha["retriever"].endswith("+veiculo")
    assert com.linha["hit@1"] >= base.linha["hit@1"]
    b = Buscador(com.itens, fabrica())
    for i, p in enumerate(ps):
        lista = b.buscar(p.texto, k=len(com.itens)).itens
        da_busca = list(zip(lista["fornecedor"], lista["codigo_interno"]))
        assert com.posicoes[i] == da_busca.index(p.pecas[0]) + 1, p.id


def test_runner_recusa_comveiculo_em_outro_campo_de_texto():
    from autopecas.evaluation.gabarito_texto import Pergunta
    from autopecas.evaluation.runner_texto import avaliar_perguntas

    df = itens().assign(descricao_norm=lambda d: d["texto_busca"])
    ps = [Pergunta("Q1", "filtro oleo gol", (("C", "3"),), "direta", "t", "", 2)]
    with pytest.raises(ValueError, match="texto_busca"):
        avaliar_perguntas(df, ps, lambda: ComVeiculo(TfidfRetriever()), nivel="Avançado", coluna_texto="descricao_norm")
