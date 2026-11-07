"""Modo Adaptacoes (src/autopecas/adaptacoes.py). Descricoes sinteticas (decisao 27)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autopecas.adaptacoes import (
    AVISO, Bloco, ROTULO_DECLARADO, ROTULO_SUGESTAO, ROTULOS_VALIDOS, sugerir,
)
from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA, normalizar_texto
from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.search import Buscador
from helpers import linha

RAIZ = Path(__file__).resolve().parents[1]


class _SemNada:
    corpus_dependente = False
    nome = "nada"

    def fit(self, textos):
        self.n = len(textos)

    def score_rows(self, linhas):
        raise NotImplementedError

    def score_texts(self, consultas):
        return np.zeros((len(consultas), self.n))


def montar(descricoes):
    """descricoes: (fornecedor, codigo, descricao, ncm, gtin)."""
    linhas = []
    for f, c, d, ncm, gtin in descricoes:
        t = normalizar_texto(d, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA)
        linhas.append(linha(f, c, d, 10.0, gtin, ncm=ncm, texto=t))
    hib = HybridRetriever(TfidfRetriever(), _SemNada(), k_rrf=1, profundidade=50, fallback_lexical=True)
    return Buscador(pd.DataFrame(linhas), hib)


def catalogo():
    d = [
        ("A", "1", "CABO CAPO UNO 85/90", "87089990", "7890000000011"),
        ("B", "2", "CABO CAPO", "87089990", None),                       # generico
        ("C", "3", "CABO CAPO UNO 95/99", "87089990", None),             # ano fora de 1988
        ("D", "4", "CABO CAPO GOL", "87089990", None),                    # outro veiculo sem relacao
        ("D", "5", "CABO CAPO UNO", "87089990", "7890000000011"),         # mesmo GTIN do item 1 (outro fornecedor)
        ("A", "6", "PAINEL CAPO UNO", "87089990", None),                  # outra peca
    ]
    # familia: gol e parati citados juntos 5x
    d += [("A", f"G{i}", "AMORTECEDOR GOL/PARATI", "87088000", None) for i in range(5)]
    d += [("B", "7", "CABO CAPO PARATI", "87089990", None)]
    return montar(d)


def codigos(bloco):
    return bloco.itens["codigo_interno"].tolist()


def bloco_de(res, trecho):
    return next((b for b in res.blocos if trecho in b.titulo), None)


def test_rotulos_sao_um_conjunto_fechado():
    assert ROTULOS_VALIDOS == {ROTULO_DECLARADO, ROTULO_SUGESTAO}
    with pytest.raises(ValueError):
        Bloco("x", "100% compatível", pd.DataFrame())


def test_citado_generico_e_aviso():
    res = sugerir(catalogo(), "cabo capo", "uno")
    cit = bloco_de(res, "Citado")
    assert set(codigos(cit)) >= {"1", "3", "5"} and cit.rotulo == ROTULO_DECLARADO
    ger = bloco_de(res, "descrição sem veículo")
    assert codigos(ger) == ["2"] and ger.rotulo == ROTULO_SUGESTAO
    assert res.aviso == AVISO and "não garante" in AVISO
    assert all(set(b.itens["rotulo"]) == {b.rotulo} for b in res.blocos)
    assert all(b.itens["motivo"].str.len().gt(0).all() for b in res.blocos)
    # outra peca que so cita o veiculo, e outro veiculo sem relacao, nao entram
    todos = {c for b in res.blocos for c in codigos(b)}
    assert "6" not in todos and "4" not in todos


def test_ano_fora_da_faixa_some_e_e_contado():
    res = sugerir(catalogo(), "cabo capo", "uno", 1988)
    cit = bloco_de(res, "Citado")
    assert "3" not in codigos(cit) and "1" in codigos(cit)
    assert res.fora_do_ano == 1
    assert "cobre 1988" in cit.itens.set_index("codigo_interno").loc["1", "motivo"]


def test_familia_vem_do_proprio_catalogo():
    res = sugerir(catalogo(), "cabo capo", "gol")
    fam = bloco_de(res, "mesma família")
    assert fam is not None and "7" in codigos(fam) and fam.rotulo == ROTULO_SUGESTAO
    assert "juntos" in fam.itens["motivo"].iloc[0]


def test_mesmo_gtin_em_outro_fornecedor():
    res = sugerir(catalogo(), "cabo capo", "uno")
    g = bloco_de(res, "mesmo GTIN")
    # os dois itens com o GTIN ja estao nos citados: nada a acrescentar
    assert g is None
    b = montar([("A", "1", "CABO CAPO UNO", "87089990", "7890000000011"),
                ("B", "2", "CB. DO CAPOT", "87089990", "7890000000011")])  # nao passa na cobertura da peca
    res = sugerir(b, "cabo capo", "uno")
    g = bloco_de(res, "mesmo GTIN")
    assert g is not None and codigos(g) == ["2"] and "7890000000011" in g.itens["motivo"].iloc[0]


def test_ncm_generico_nao_e_categoria():
    res = sugerir(catalogo(), "cabo capo", "uno")
    assert "NCM" not in " ".join(bloco_de(res, "descrição sem veículo").itens["motivo"])
    assert bloco_de(res, "relacionadas") is None  # NCM 87089990 e generico: nao agrupa nada
    b = montar([
        ("A", "1", "CABO CAPO UNO", "87089910", None),
        ("B", "2", "CABO CAPO", "87089910", None),
        ("C", "3", "MOLA CAPO", "87089910", None),
    ])
    res = sugerir(b, "cabo capo", "uno")
    assert "mesmo NCM 87089910" in bloco_de(res, "descrição sem veículo").itens["motivo"].iloc[0]
    rel = bloco_de(res, "relacionadas")
    assert rel is None or rel.rotulo == ROTULO_SUGESTAO  # peca relacionada, quando aparece, e so sugestao


def test_sem_veiculo_ou_sem_peca_devolve_vazio():
    b = catalogo()
    assert sugerir(b, "cabo capo", None).total() == 0
    assert sugerir(b, "uno", "uno").total() == 0   # so o veiculo: nao ha peca


def test_veiculo_e_ano_podem_vir_na_propria_peca():
    res = sugerir(catalogo(), "cabo capo uno 1988")
    assert res.veiculo == "uno" and res.ano == 1988 and res.fora_do_ano == 1


def test_nenhum_texto_do_produto_promete_100_por_cento_compativel():
    """Regra de produto: nunca 'compativel' absoluto. Varre src/ e app/ (testes podem citar a frase)."""
    padrao = re.compile(r"100\s*%[^\n]{0,40}compat|compat[^\n]{0,40}100\s*%", re.I)
    achados = []
    for pasta in ("src", "app"):
        for f in (RAIZ / pasta).rglob("*.py"):
            for n, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if padrao.search(ln) and "nunca" not in ln.lower() and "nao " not in ln.lower():
                    achados.append(f"{f.relative_to(RAIZ)}:{n}: {ln.strip()}")
    assert achados == []
