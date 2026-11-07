"""Diagnostico da hipotese da Etapa 4: o que fica a frente da 1a peca esperada?

Hipotese (CONTEXTO, 5.5): "o nome do veiculo puxa itens de OUTRO veiculo para antes do correto". Para cada
pergunta com resposta em que a 1a peca esperada nao e a 1a do ranking, classifica os itens que ficaram a
frente pelo que a DESCRICAO deles cita, comparado com o veiculo da pergunta:

- `cita_o_veiculo`   cita o veiculo da pergunta (pode ser outra peca)
- `sem_veiculo`      generico: a descricao nao cita veiculo nenhum
- `outro_veiculo`    cita so outro veiculo (a hipotese)
- `consulta_sem_veiculo`  a pergunta nao cita veiculo conhecido (a classificacao nao se aplica)

Classificacao automatica e so de veiculo. Se o item a frente e OUTRA PECA ou uma peca valida que o
gabarito esqueceu, so a leitura humana do `revisao_*.md` diz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autopecas.evaluation.metrics import DECIMAIS_EMPATE
from autopecas.evaluation.runner_texto import ResultadoTexto
from autopecas.retrieval.veiculo import CAMADA_CITA, CAMADA_NEUTRA, CAMADA_OUTRO, garantir_extracao, situacao_do_item
from autopecas.veiculos import analisar_consulta

COLUNAS = ["id", "tipo", "pergunta", "posicao_1a_esperada", "a_frente", "cita_o_veiculo",
           "sem_veiculo", "outro_veiculo", "consulta_sem_veiculo"]


def itens_a_frente(res: ResultadoTexto) -> pd.DataFrame:
    itens = garantir_extracao(res.itens).reset_index(drop=True)
    veic, ini, fim = itens["veiculos"].to_numpy(), itens["ano_ini"].to_numpy(), itens["ano_fim"].to_numpy()
    s = np.round(res.scores, DECIMAIS_EMPATE)
    linhas = []
    for i, id_ in enumerate(res.gabarito.ids):
        rel = set(res.gabarito.relevantes[i].tolist())
        if not rel:
            continue
        ordem = np.argsort(-s[i], kind="stable")
        ordem = ordem[s[i][ordem] > 0]
        k = next((n for n, q in enumerate(ordem) if q in rel), len(ordem))
        if k == 0:
            continue
        ahead = ordem[:k]
        consulta = analisar_consulta(res.gabarito.perguntas[i])
        cont = {"cita_o_veiculo": 0, "sem_veiculo": 0, "outro_veiculo": 0, "consulta_sem_veiculo": 0}
        for q in ahead:
            if not consulta.tem_veiculo:
                cont["consulta_sem_veiculo"] += 1
                continue
            camada, _ = situacao_do_item(veic[q], ini[q], fim[q], consulta)
            cont["cita_o_veiculo" if camada == CAMADA_CITA else "sem_veiculo" if camada == CAMADA_NEUTRA
                 else "outro_veiculo"] += 1
        linhas.append({"id": id_, "tipo": res.gabarito.tipos[i], "pergunta": res.gabarito.perguntas[i],
                       "posicao_1a_esperada": k + 1, "a_frente": len(ahead), **cont})
    return pd.DataFrame(linhas, columns=COLUNAS)


def resumo(tabela: pd.DataFrame) -> dict:
    """Totais de itens a frente por classe, so nas perguntas que citam veiculo."""
    com = tabela[tabela["consulta_sem_veiculo"] == 0]
    return {
        "perguntas_erradas": int(len(tabela)),
        "perguntas_com_veiculo": int(len(com)),
        "itens_a_frente": int(com["a_frente"].sum()),
        "cita_o_veiculo": int(com["cita_o_veiculo"].sum()),
        "sem_veiculo": int(com["sem_veiculo"].sum()),
        "outro_veiculo": int(com["outro_veiculo"].sum()),
        "perguntas_com_algum_outro_veiculo": int((com["outro_veiculo"] > 0).sum()),
    }
