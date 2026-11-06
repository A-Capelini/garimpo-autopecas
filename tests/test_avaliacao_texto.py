import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import df_de, linha

from autopecas.evaluation.gabarito_texto import ErroGabarito, Pergunta, ler_gabarito
from autopecas.evaluation.metrics import (
    avaliar_texto,
    ic95_bootstrap,
    posicoes_das_relevantes,
    primeira_posicao_texto,
)
from autopecas.evaluation.runner_texto import (
    COLUNAS_TEXTO,
    acrescentar_csv,
    avaliar_perguntas,
    comparar_pareado,
    normalizar_pergunta,
    relatorio_revisao,
    tabela_de,
    tabela_legivel,
)
from autopecas.retrieval.hybrid import HybridRetriever
from autopecas.retrieval.lexical import TfidfRetriever
from autopecas.search import Buscador

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_avaliar.py"


# ---------------------------------------------------------------- metricas
def test_posicoes_sem_excluir_nenhuma_linha_e_sem_evidencia_vale_n_mais_1():
    # 4 documentos; pergunta 0: relevante = doc 0 (1o lugar, nao e excluido); pergunta 1: doc 3 com score 0
    scores = np.array([[0.9, 0.5, 0.1, 0.0], [0.9, 0.5, 0.1, 0.0]])
    p = posicoes_das_relevantes(scores, [np.array([0, 2]), np.array([3])])
    assert p[0].tolist() == [1, 3]
    assert p[1].tolist() == [5]  # n_docs + 1: sem evidencia nao aparece na lista


def test_empate_com_ruido_desempata_pela_posicao_do_documento():
    scores = np.array([[0.5 + 1e-16, 0.5, 0.5 - 1e-16]])
    assert posicoes_das_relevantes(scores, [np.array([1])])[0].tolist() == [2]


def test_avaliar_texto_hit_mrr_recall_com_varias_pecas_esperadas():
    scores = np.array([
        [0.9, 0.8, 0.1, 0.1, 0.1, 0.1],   # relevantes {0, 1}: posicoes 1 e 2
        [0.1, 0.1, 0.9, 0.8, 0.7, 0.6],   # relevante {5}: posicao 4 (os empates em 0.1 vem depois)
    ])
    rel = [np.array([0, 1]), np.array([5])]
    m = avaliar_texto(scores, rel)
    assert m["n_consultas"] == 2
    assert m["hit@1"] == 0.5 and m["hit@5"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 1 / 4) / 2)
    assert m["recall@1"] == pytest.approx((0.5 + 0) / 2)
    assert m["recall@10"] == 1.0
    assert primeira_posicao_texto(scores, rel).tolist() == [1, 4]


def test_avaliar_texto_sem_perguntas_devolve_nan_e_pergunta_sem_peca_e_erro():
    assert np.isnan(avaliar_texto(np.empty((0, 3)), [])["mrr"])
    with pytest.raises(ValueError, match="sem peca esperada"):
        primeira_posicao_texto(np.ones((1, 3)), [np.array([], dtype=int)])
    with pytest.raises(ValueError, match="mesmo numero"):
        posicoes_das_relevantes(np.ones((2, 3)), [np.array([0])])


def test_bootstrap_e_reprodutivel_contem_a_media_e_degenera_sem_variacao():
    v = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0], dtype=float)
    lo, hi = ic95_bootstrap(v)
    assert (lo, hi) == ic95_bootstrap(v)  # semente fixa
    assert lo <= v.mean() <= hi and 0 <= lo < hi <= 1
    assert ic95_bootstrap(np.ones(20)) == (1.0, 1.0)
    assert np.isnan(ic95_bootstrap(np.array([]))[0])


def test_comparar_pareado_ganho_perda_e_mesmo_vetor():
    a = np.array([1, 1, 2, 1, 3])
    b = np.array([2, 1, 1, 4, 3])
    c = comparar_pareado(a, b)
    assert c["delta_hit@1"] == pytest.approx((3 - 2) / 5)
    assert c["delta_mrr"] == pytest.approx(np.mean(1 / a - 1 / b))
    lo, hi = c["ic95_hit@1"]
    assert lo <= c["delta_hit@1"] <= hi
    zero = comparar_pareado(a, a)
    assert zero["delta_mrr"] == 0 and zero["ic95_mrr"] == (0.0, 0.0)
    with pytest.raises(ValueError):
        comparar_pareado(a, b[:3])


# ---------------------------------------------------------------- runner
def catalogo():
    itens = [
        ("A", "1", "cabo capo gol 1740mm"), ("A", "2", "filtro oleo uno"), ("A", "3", "vela ignicao palio"),
        ("B", "1", "cabo do capo gol/saveiro"), ("B", "2", "amortecedor dianteiro fusca"),
        ("C", "1", "pastilha freio uno"), ("C", "2", "cabo acelerador fusca"),
        ("D", "1", "kit embreagem fiesta"), ("D", "2", "filtro de oleo palio"),
        ("E", "1", "cabo capo gol especial"),  # fora do Basico: nao pode aparecer
    ]
    return df_de(*[linha(f, c, d, texto=d.replace("/", " ")) for f, c, d in itens])


def perguntas():
    return [
        Pergunta("Q1", "cabo do capô do gol", (("A", "1"), ("B", "1")), "direta", "", "", 2),
        Pergunta("Q2", "amortecedor fusca", (("B", "2"),), "coloquial", "", "", 3),
        Pergunta("Q3", "filtro de óleo palio", (("D", "2"),), "direta", "", "", 4),
        Pergunta("Q4", "turbina de avião", (), "sem_resposta", "", "", 5),
    ]


class Espiao(TfidfRetriever):
    tamanhos: list[int] = []

    def fit(self, textos):
        Espiao.tamanhos.append(len(list(textos)))
        super().fit(textos)


def test_runner_indexa_so_o_nivel_e_exclui_sem_resposta_das_metricas():
    Espiao.tamanhos = []
    res = avaliar_perguntas(catalogo(), perguntas(), Espiao)
    assert Espiao.tamanhos == [9]  # 10 itens menos o do fornecedor E
    assert len(res.itens) == 9 and "E" not in set(res.itens["fornecedor"])
    assert res.linha["n_docs"] == 9 and res.linha["n_consultas"] == 3 and res.linha["n_sem_resposta"] == 1
    assert res.ids_medidos == ("Q1", "Q2", "Q3") and len(res.posicoes) == 3
    assert res.scores.shape == (4, 9)
    assert 0 <= res.linha["hit@1_ic_inf"] <= res.linha["hit@1"] <= res.linha["hit@1_ic_sup"] <= 1
    assert list(tabela_de([res]).columns) == COLUNAS_TEXTO


def test_runner_acerta_as_perguntas_obvias():
    res = avaliar_perguntas(catalogo(), perguntas(), TfidfRetriever)
    assert res.posicoes[1] == 1  # amortecedor fusca -> B:2
    assert res.linha["hit@5"] == 1.0


def test_normalizacao_da_pergunta_e_a_do_buscador():
    assert normalizar_pergunta("Cabo do CAPÔ — Gol/Saveiro") == "cabo do capo gol saveiro"


class EmbTrigramas:
    """Vetores deterministicos por texto (hash de trigramas), com score_texts; cosseno nao depende do corpus."""

    nome = "emb-falso"
    corpus_dependente = False

    def _vec(self, textos):
        E = np.zeros((len(textos), 64))
        for i, t in enumerate(textos):
            for k in range(len(t) - 2):
                E[i, sum(ord(c) * (n + 1) for n, c in enumerate(t[k : k + 3])) % 64] += 1
        n = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.where(n == 0, 1, n)

    def fit(self, textos):
        self._E = self._vec(list(textos))

    def score_rows(self, linhas):
        return self._E[list(linhas)] @ self._E.T

    def score_texts(self, consultas):
        return self._vec(list(consultas)) @ self._E.T


def _hibrido():
    return HybridRetriever(TfidfRetriever(), EmbTrigramas(), metodo="rrf", peso_lexical=3.0,
                           peso_semantico=1.0, k_rrf=1, profundidade=50, fallback_lexical=True)


def test_medicao_equivale_ao_que_o_buscador_devolve():
    """A avaliacao mede o mesmo ranking que o app mostra (mesma normalizacao, fusao e desempate)."""
    df, ps = catalogo(), perguntas()[:3]
    res = avaliar_perguntas(df, ps, _hibrido)
    buscador = Buscador(res.itens, _hibrido())
    for i, p in enumerate(ps):
        lista = buscador.buscar(p.texto, k=len(res.itens)).itens
        da_busca = [(f, c) for f, c in zip(lista["fornecedor"], lista["codigo_interno"])]
        esperado = min(da_busca.index(peca) + 1 for peca in p.pecas if peca in da_busca)
        assert res.posicoes[i] == esperado, p.id


def test_runner_recusa_gabarito_que_nao_bate_com_o_catalogo():
    ruins = [Pergunta("Q1", "x", (("A", "999"),), "direta", "", "", 2)]
    with pytest.raises(ErroGabarito):
        avaliar_perguntas(catalogo(), ruins, TfidfRetriever)


def test_tabela_legivel_mostra_o_intervalo():
    res = avaliar_perguntas(catalogo(), perguntas(), TfidfRetriever)
    leg = tabela_legivel(tabela_de([res]))
    assert "[" in leg.loc[0, "hit@1"] and leg.loc[0, "perguntas"] == 3


def test_relatorio_de_revisao_marca_pecas_do_gabarito_e_lista_sem_resposta():
    res = avaliar_perguntas(catalogo(), perguntas(), TfidfRetriever)
    md = relatorio_revisao(res, perguntas())
    assert "## Q1 — cabo do capô do gol" in md and "esperadas: A:1 → posição" in md
    assert "✅ B:2" in md  # a peca esperada de Q2 aparece marcada
    q4 = md.split("## Q4")[1]
    assert "esperadas" not in q4 and "tipo: sem_resposta" in q4  # sem_resposta: so o que foi devolvido


def test_acrescentar_csv_recusa_cabecalho_diferente(tmp_path):
    res = avaliar_perguntas(catalogo(), perguntas(), TfidfRetriever)
    tab = tabela_de([res])
    destino = tmp_path / "pasta" / "avaliacao_texto.csv"
    acrescentar_csv(tab, destino)
    acrescentar_csv(tab, destino)
    lido = pd.read_csv(destino)
    assert len(lido) == 2 and list(lido.columns) == COLUNAS_TEXTO
    destino.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colunas diferentes"):
        acrescentar_csv(tab, destino)


# ---------------------------------------------------------------- script 02 (modo --gabarito)
def _catalogo_csv(tmp_path):
    df = catalogo()[["codigo_interno", "descricao", "preco", "ncm", "fornecedor"]].copy()
    df["gtin"] = None
    caminho = tmp_path / "catalogo.csv"
    df.to_csv(caminho, index=False)
    return caminho


def _gabarito_csv(tmp_path, texto=None):
    caminho = tmp_path / "gabarito.csv"
    caminho.write_text(
        texto or "id;pergunta;pecas;tipo\nQ1;cabo do capô do gol;A:1|B:1;direta\n"
        "Q2;amortecedor fusca;B:2;coloquial\nQ3;pastilha freio uno;C:1;direta\nQ4;turbina de avião;;sem_resposta\n",
        encoding="utf-8",
    )
    return caminho


def _rodar(monkeypatch, *argumentos):
    monkeypatch.setattr(sys, "argv", ["02_avaliar.py", *map(str, argumentos)])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_script_so_validar_nao_gera_arquivos(tmp_path, monkeypatch, capsys):
    _rodar(monkeypatch, "--retriever", "lexical", "--gabarito", _gabarito_csv(tmp_path), "--so-validar",
           "--catalogo", _catalogo_csv(tmp_path), "--resultados", tmp_path / "res")
    saida = capsys.readouterr().out
    assert "Gabarito OK: 4 perguntas (3 com resposta, 1 sem resposta)" in saida
    assert "AVISO" in saida  # menos de 30 perguntas
    assert not (tmp_path / "res").exists()


def test_script_mede_lexical_no_basico_e_grava_csv_e_revisao(tmp_path, monkeypatch, capsys):
    _rodar(monkeypatch, "--retriever", "lexical", "--gabarito", _gabarito_csv(tmp_path),
           "--catalogo", _catalogo_csv(tmp_path), "--resultados", tmp_path / "res")
    saida = capsys.readouterr().out
    assert "Nível Básico: 9 itens; 3 perguntas medidas" in saida
    lido = pd.read_csv(tmp_path / "res" / "avaliacao_texto.csv")
    assert len(lido) == 1 and lido.loc[0, "nivel"] == "Básico" and lido.loc[0, "n_consultas"] == 3
    assert lido.loc[0, "normalizacao"] == "simples" and lido.loc[0, "campo"] == "texto_busca"
    import hashlib

    assert lido.loc[0, "gabarito_sha256"] == hashlib.sha256((tmp_path / "gabarito.csv").read_bytes()).hexdigest()[:12]
    assert (tmp_path / "res" / "revisao_basico_tfidf-char.md").exists()
    assert not (tmp_path / "res" / "avaliacao.csv").exists()  # o gabarito por GTIN nao e tocado


def test_script_gabarito_invalido_encerra_com_mensagem_clara(tmp_path, monkeypatch):
    ruim = _gabarito_csv(tmp_path, "id;pergunta;pecas\nQ1;x;A:999\n")
    with pytest.raises(SystemExit) as e:
        _rodar(monkeypatch, "--retriever", "lexical", "--gabarito", ruim,
               "--catalogo", _catalogo_csv(tmp_path), "--resultados", tmp_path / "res")
    assert "Gabarito inválido" in str(e.value) and "código não existe" in str(e.value)


def test_script_todos_e_so_validar_exigem_gabarito(tmp_path, monkeypatch):
    for extra in (["--retriever", "todos"], ["--retriever", "lexical", "--so-validar"]):
        with pytest.raises(SystemExit, match="só existe com --gabarito"):
            _rodar(monkeypatch, *extra, "--catalogo", _catalogo_csv(tmp_path), "--resultados", tmp_path / "res")


def test_script_preset_escolhido_aplica_as_decisoes_21_a_23(monkeypatch):
    modulo = runpy.run_path(str(SCRIPT), run_name="garimpo_02")
    import argparse

    a = argparse.Namespace(preset="escolhido", k_rrf=60, min_df=2, normalizacao="oficial", metodo="soma",
                           peso_lexical=0.5, peso_semantico=2.0, profundidade=10, sem_sublinear=True)
    modulo["_aplicar_preset"](a)
    assert (a.k_rrf, a.min_df, a.normalizacao, a.metodo) == (1, 1, "simples", "rrf")
    assert (a.peso_lexical, a.peso_semantico, a.profundidade, a.sem_sublinear) == (3.0, 1.0, 50, False)


def test_script_todos_mede_lexical_semantico_e_hibrido_e_imprime_diferenca_pareada(
    tmp_path, monkeypatch, capsys
):
    class SemanticoFalso(EmbTrigramas):  # no lugar do Ollama: sem rede, sem servidor
        def __init__(self, modelo, prefixo_documento="", prefixo_consulta="", **_):
            self.nome = f"ollama:{modelo}"

    monkeypatch.setattr("autopecas.retrieval.ollama_embed.OllamaEmbeddingRetriever", SemanticoFalso)
    _rodar(monkeypatch, "--retriever", "todos", "--gabarito", _gabarito_csv(tmp_path),
           "--catalogo", _catalogo_csv(tmp_path), "--resultados", tmp_path / "res")
    saida = capsys.readouterr().out
    assert "Diferença pareada híbrido − léxico" in saida and "Δ hit@1" in saida and "Δ mrr" in saida
    lido = pd.read_csv(tmp_path / "res" / "avaliacao_texto.csv")
    assert lido["retriever"].tolist() == ["tfidf-char", "ollama:bge-m3", "hibrido-rrf[3:1,k=1,d=50]"]
    assert (lido["n_consultas"] == 3).all() and (lido["normalizacao"] == "simples").all()
    assert len(list((tmp_path / "res").glob("revisao_basico_*.md"))) == 3
