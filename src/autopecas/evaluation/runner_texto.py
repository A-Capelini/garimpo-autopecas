"""Avaliacao de recuperadores com perguntas em texto livre (gabarito manual) em UM nivel.

Diferencas em relacao a `runner.avaliar_niveis` (consulta = item do catalogo):
- a consulta e o texto da pergunta, normalizado como o texto de busca (mesma regra do Buscador);
- o indice e o do nivel: lexical reajustado so com os itens do nivel; embeddings so dos itens do nivel
  (o cosseno nao depende do resto do corpus, entao o resultado e o mesmo de indexar tudo);
- perguntas `sem_resposta` ficam fora das metricas (so aparecem na revisao manual);
- toda metrica vem com IC95% por bootstrap sobre as perguntas: com 30 a 50 perguntas o intervalo e
  largo, e diferenca pequena entre recuperadores nao e conclusiva.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA, normalizar_texto
from autopecas.evaluation.gabarito_texto import GabaritoTexto, Pergunta, itens_do_nivel, resolver
from autopecas.evaluation.metrics import (
    DECIMAIS_EMPATE,
    avaliar_texto,
    ic95_bootstrap,
    posicoes_das_relevantes,
    primeira_posicao_texto,
)

METRICAS_COM_IC = ("hit@1", "hit@5", "hit@10", "mrr")
COLUNAS_TEXTO = (
    ["retriever", "nivel", "n_docs", "n_consultas", "n_sem_resposta",
     "hit@1", "hit@5", "hit@10", "recall@10", "mrr", "mrr@10"]
    + [f"{m}_ic_{lado}" for m in METRICAS_COM_IC for lado in ("inf", "sup")]
    + ["seg_indexacao", "ms_por_consulta"]
)
K_REVISAO = 5


def normalizar_pergunta(texto: str) -> str:
    """Mesma normalizacao do Buscador (decisao 21)."""
    return normalizar_texto(texto, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA)


@dataclass(frozen=True)
class ResultadoTexto:
    linha: dict  # uma linha de COLUNAS_TEXTO
    gabarito: GabaritoTexto
    itens: pd.DataFrame  # itens do nivel (as posicoes do gabarito e do ranking valem para ele)
    scores: np.ndarray  # perguntas x itens (todas as perguntas, inclusive sem_resposta)
    posicoes: np.ndarray  # 1o equivalente, so das perguntas COM resposta (mesma ordem de ids_medidos)
    ids_medidos: tuple[str, ...]


def avaliar_perguntas(
    df: pd.DataFrame,
    perguntas: list[Pergunta],
    fabrica: Callable[[], object],
    nivel: str = "Básico",
    coluna_texto: str = "texto_busca",
    normalizar: Callable[[str], str] = normalizar_pergunta,
    n_boot: int = 2000,
) -> ResultadoTexto:
    """Mede um recuperador (`fabrica()` devolve um novo, sem ajuste) nas perguntas do gabarito."""
    sub = itens_do_nivel(df, nivel)
    gab = resolver(perguntas, df, nivel)
    r = fabrica()
    t0 = time.time()
    if hasattr(r, "fit_itens"):  # recuperador que precisa dos campos do item (ex.: ComVeiculo)
        if coluna_texto != "texto_busca":
            raise ValueError("a ordenação por veículo só vale sobre o campo texto_busca")
        r.fit_itens(sub)
    else:
        r.fit(sub[coluna_texto].tolist())
    t_fit = time.time() - t0

    t1 = time.time()
    scores = np.asarray(r.score_texts([normalizar(t) for t in gab.perguntas]), dtype=np.float64)
    ms = 1000 * (time.time() - t1) / max(len(gab), 1)

    com = gab.com_resposta
    rel = [rel for rel, c in zip(gab.relevantes, com) if c]
    pos = primeira_posicao_texto(scores[com], rel) if com.any() else np.array([], dtype=int)
    m = avaliar_texto(scores[com], rel)
    ic = {
        "hit@1": ic95_bootstrap((pos == 1).astype(float), n_boot),
        "hit@5": ic95_bootstrap((pos <= 5).astype(float), n_boot),
        "hit@10": ic95_bootstrap((pos <= 10).astype(float), n_boot),
        "mrr": ic95_bootstrap(1.0 / pos if len(pos) else pos, n_boot),
    }
    linha = {
        "retriever": r.nome, "nivel": nivel, "n_docs": len(sub),
        "n_consultas": int(com.sum()), "n_sem_resposta": int((~com).sum()),
        **{c: m[c] for c in ("hit@1", "hit@5", "hit@10", "recall@10", "mrr", "mrr@10")},
        **{f"{k}_ic_{lado}": v for k, (lo, hi) in ic.items() for lado, v in (("inf", lo), ("sup", hi))},
        "seg_indexacao": round(t_fit, 1), "ms_por_consulta": round(ms, 2),
    }
    ids = tuple(i for i, c in zip(gab.ids, com) if c)
    return ResultadoTexto(linha, gab, sub, scores, pos, ids)


def tabela_de(resultados: list[ResultadoTexto]) -> pd.DataFrame:
    return pd.DataFrame([r.linha for r in resultados])[COLUNAS_TEXTO]


def tabela_legivel(tabela: pd.DataFrame) -> pd.DataFrame:
    """Versao para imprimir: 'valor [inf; sup]' nas metricas com IC."""
    out = pd.DataFrame({"retriever": tabela["retriever"], "perguntas": tabela["n_consultas"]})
    for m in METRICAS_COM_IC:
        out[m] = [f"{v:.3f} [{lo:.2f}; {hi:.2f}]"
                  for v, lo, hi in zip(tabela[m], tabela[f"{m}_ic_inf"], tabela[f"{m}_ic_sup"])]
    out["mrr@10"] = tabela["mrr@10"].map(lambda v: f"{v:.3f}")
    out["recall@10"] = tabela["recall@10"].map(lambda v: f"{v:.3f}")
    return out


def comparar_pareado(pos_a: np.ndarray, pos_b: np.ndarray, n_boot: int = 2000) -> dict:
    """Diferenca A - B nas MESMAS perguntas (posicoes do 1o equivalente), com IC95% pareado."""
    pos_a, pos_b = np.asarray(pos_a), np.asarray(pos_b)
    if pos_a.shape != pos_b.shape:
        raise ValueError("as duas medicoes precisam ser das mesmas perguntas")
    d = {
        "hit@1": (pos_a == 1).astype(float) - (pos_b == 1).astype(float),
        "hit@5": (pos_a <= 5).astype(float) - (pos_b <= 5).astype(float),
        "mrr": (1.0 / pos_a - 1.0 / pos_b) if len(pos_a) else pos_a.astype(float),
    }
    saida = {}
    for k, v in d.items():
        saida[f"delta_{k}"] = float(v.mean()) if len(v) else float("nan")
        saida[f"ic95_{k}"] = ic95_bootstrap(v, n_boot)
    return saida


def posicoes_por_pergunta(resultados: list[ResultadoTexto]) -> pd.DataFrame:
    """Posicao do 1o equivalente de cada pergunta COM resposta, uma coluna por recuperador.

    Todos os resultados precisam ser do mesmo gabarito. Posicao n_docs+1 = fora da lista (sem evidencia).
    """
    base = resultados[0]
    for r in resultados[1:]:
        if r.ids_medidos != base.ids_medidos:
            raise ValueError("os resultados precisam ser do mesmo gabarito")
    com = base.gabarito.com_resposta
    tab = pd.DataFrame({
        "id": base.ids_medidos,
        "pergunta": [t for t, c in zip(base.gabarito.perguntas, com) if c],
        "tipo": [t for t, c in zip(base.gabarito.tipos, com) if c],
    })
    for r in resultados:
        tab[r.linha["retriever"]] = r.posicoes
    return tab


def metricas_por_tipo(resultados: list[ResultadoTexto]) -> pd.DataFrame:
    """Hit@1/5/10 e MRR por tipo de pergunta e recuperador. Descritivo: n por tipo e pequeno (sem IC)."""
    pos = posicoes_por_pergunta(resultados)
    linhas = []
    for r in resultados:
        nome = r.linha["retriever"]
        for tipo, grupo in pos.groupby("tipo", sort=True):
            p = grupo[nome].to_numpy()
            linhas.append({
                "retriever": nome, "nivel": r.linha["nivel"], "tipo": tipo, "n_consultas": len(p),
                "hit@1": float(np.mean(p == 1)), "hit@5": float(np.mean(p <= 5)),
                "hit@10": float(np.mean(p <= 10)), "mrr": float(np.mean(1.0 / p)),
            })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- arquivos
def acrescentar_csv(tabela: pd.DataFrame, destino: Path | str) -> None:
    """Acrescenta ao CSV; se o arquivo ja existe com outro cabecalho, recusa (nao mistura colunas)."""
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 0:
        with destino.open(newline="", encoding="utf-8") as f:
            existente = next(csv.reader(f), [])
        if existente != list(tabela.columns):
            raise ValueError(
                f"{destino} já existe com colunas diferentes; mova-o (ex.: renomeie) antes de continuar."
            )
        tabela.to_csv(destino, mode="a", header=False, index=False)
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(destino, mode="w", header=True, index=False)


def relatorio_revisao(res: ResultadoTexto, perguntas: list[Pergunta], k: int = K_REVISAO) -> str:
    """Markdown para o grupo revisar: por pergunta, onde ficaram as pecas esperadas e o top-k.

    Serve para achar peca que serve e nao foi marcada (gabarito incompleto) e para ver o que as
    perguntas sem_resposta devolvem. Contem descricoes e precos: vai para resultados/ (fora do Git).
    """
    gab, itens = res.gabarito, res.itens
    por_id = {p.id: p for p in perguntas}
    s = np.round(res.scores, DECIMAIS_EMPATE)
    esperadas = posicoes_das_relevantes(res.scores, gab.relevantes)
    linhas = [
        f"# Revisão do gabarito — {res.linha['retriever']} — nível {gab.nivel}",
        "",
        f"{res.linha['n_consultas']} pergunta(s) com resposta, {res.linha['n_sem_resposta']} sem resposta. "
        "✅ = peça do gabarito. Posição n_docs+1 = fora da lista (sem evidência).",
        "",
    ]
    for i, id_ in enumerate(gab.ids):
        p = por_id[id_]
        linhas += [f"## {id_} — {p.texto}", f"tipo: {p.tipo}" + (f" · autor: {p.autor}" if p.autor else ""), ""]
        if len(gab.relevantes[i]):
            linhas.append("esperadas: " + "; ".join(
                f"{itens.at[q, 'fornecedor']}:{itens.at[q, 'codigo_interno']} → posição {pos}"
                for q, pos in zip(gab.relevantes[i], esperadas[i])
            ))
            linhas.append("")
        marcadas = set(gab.relevantes[i].tolist())
        ordem = np.argsort(-s[i], kind="stable")
        ordem = ordem[s[i][ordem] > 0][:k]
        if len(ordem) == 0:
            linhas.append("(nada devolvido: nenhum item com evidência)")
        for r, q in enumerate(ordem, 1):
            marca = "✅" if q in marcadas else "  "
            linhas.append(
                f"{r}. {marca} {itens.at[q, 'fornecedor']}:{itens.at[q, 'codigo_interno']} · "
                f"{itens.at[q, 'descricao']} · score {res.scores[i, q]:.3f}"
            )
        linhas.append("")
    return "\n".join(linhas)
