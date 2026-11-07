"""Ordenacao por veiculo (Etapa 4): o veiculo da consulta vira SINAL separado, nao so mais um termo.

Problema medido (CONTEXTO, secao 5.5): no Basico, nas perguntas `direta` e `coloquial`, o item certo
(fornecedores A e C, que citam o veiculo e abreviam a peca: "PAST. FREIO DIANT. CORSA") fica atras de
itens genericos de B e D que escrevem a peca por extenso ("MOLA DA PASTILHA DO FREIO DIANTEIRO").
O TF-IDF de caracteres da pouco peso a um nome curto como "uno" ou "gol".

Regra (sem parametro novo; a profundidade e a mesma do RRF): so se a consulta cita um veiculo, os
`PROFUNDIDADE` primeiros itens do ranking-base sao reordenados por camada, mantendo a ordem-base
dentro de cada camada; o resto do ranking nao muda.

  camada 1  cita o veiculo da consulta (e o ano cabe na faixa, se houver) E cobre a maioria das
            palavras da peca;
  camada 2  neutro: sem veiculo na descricao (generico), ou cita o veiculo mas e outra peca;
  camada 3  cita so outro veiculo, ou cita o veiculo com a faixa de ano fora do pedido.

A exigencia de cobertura evita promover a peca errada que so cita o veiculo (ex.: "MANGUEIRA DO
FILTRO DE AR - GOL" para a consulta "filtro oleo gol"). A cobertura aceita abreviacao por prefixo
(ACEL. = acelerador) so como criterio de elegibilidade: o texto de busca e os vetores nao mudam.

Limites conhecidos: siglas de 2 letras ("CB." = cabo) nao casam; peca relacionada com as mesmas palavras
("MANGUEIRA DO FILTRO DE OLEO") ainda passa; so vale quando o veiculo esta no vocabulario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autopecas.data.quality import normalizar_texto
from autopecas.veiculos import ConsultaAnalisada, analisar_consulta, aplicar_extracao

PROFUNDIDADE = 50
COBERTURA_MINIMA = 0.5  # estritamente acima: a MAIORIA das palavras da peca

CAMADA_CITA, CAMADA_NEUTRA, CAMADA_OUTRO = 1, 2, 3
DECIMAIS_EMPATE = 10  # mesmo criterio do Buscador e das metricas

_STOP = {"de", "do", "da", "dos", "das", "e", "a", "o", "para", "p", "com", "c", "s", "sem",
         "no", "na", "em", "ao", "um", "uma", "os", "as"}


def palavras_da_peca(texto: str) -> list[str]:
    return [w for w in texto.split() if w not in _STOP and len(w) > 1]


def casa(a: str, b: str) -> bool:
    """Mesma palavra ou abreviacao: prefixo comum de pelo menos 3 letras e 80% da menor palavra."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 3 and n >= 0.8 * min(len(a), len(b))


def cobertura(palavras_peca: list[str], texto_item: str) -> float:
    """Fracao das palavras da peca que aparecem no item (igual ou abreviada). Sem palavras = 1."""
    if not palavras_peca:
        return 1.0
    palavras_item = texto_item.split()
    return sum(any(casa(q, i) for i in palavras_item) for q in palavras_peca) / len(palavras_peca)


def _faixa_ok(ano: int | None, ini, fim) -> bool:
    if ano is None or (pd.isna(ini) and pd.isna(fim)):
        return True  # sem ano pedido, ou item sem ano informado: nao exclui
    return (pd.isna(ini) or ini <= ano) and (pd.isna(fim) or ano <= fim)


def situacao_do_item(veiculos_item, ano_ini, ano_fim, consulta: ConsultaAnalisada) -> tuple[int, str]:
    """(camada descritiva, texto para a tela) SEM o criterio de cobertura. Vazio se a consulta nao cita veiculo."""
    if not consulta.tem_veiculo:
        return CAMADA_CITA, ""
    if not veiculos_item:
        return CAMADA_NEUTRA, "sem veículo na descrição (genérico)"
    comuns = [v for v in veiculos_item if v in consulta.veiculos]
    if not comuns:
        return CAMADA_OUTRO, "cita outro veículo: " + ", ".join(veiculos_item)
    if not _faixa_ok(consulta.ano, ano_ini, ano_fim):
        return CAMADA_OUTRO, f"cita {comuns[0]}, mas o ano {consulta.ano} está fora da faixa"
    return CAMADA_CITA, f"cita {', '.join(comuns)}"


def camadas_para_ordenar(itens: pd.DataFrame, posicoes: np.ndarray, consulta: ConsultaAnalisada) -> np.ndarray:
    """Camada (1, 2 ou 3) de cada posicao em `posicoes` (indices de linha de `itens`), com cobertura."""
    palavras = palavras_da_peca(consulta.peca)
    veiculos, ini, fim, texto = (itens[c].to_numpy() for c in ("veiculos", "ano_ini", "ano_fim", "texto_busca"))
    saida = np.empty(len(posicoes), dtype=int)
    for k, p in enumerate(posicoes):
        camada, _ = situacao_do_item(veiculos[p], ini[p], fim[p], consulta)
        if camada == CAMADA_CITA and cobertura(palavras, texto[p]) <= COBERTURA_MINIMA:
            camada = CAMADA_NEUTRA
        saida[k] = camada
    return saida


def reordenar(
    ordem: np.ndarray, itens: pd.DataFrame, consulta: ConsultaAnalisada, profundidade: int = PROFUNDIDADE
) -> np.ndarray:
    """`ordem`: posicoes em `itens`, da mais para a menos relevante (so itens com evidencia).

    Devolve a nova ordem. Sem veiculo na consulta, devolve `ordem` como veio.
    """
    if not consulta.tem_veiculo or len(ordem) == 0:
        return ordem
    topo, resto = ordem[:profundidade], ordem[profundidade:]
    camadas = camadas_para_ordenar(itens, topo, consulta)
    topo = topo[np.argsort(camadas, kind="stable")]
    return np.concatenate([topo, resto])


def garantir_extracao(itens: pd.DataFrame) -> pd.DataFrame:
    return itens if {"veiculos", "ano_ini", "ano_fim"} <= set(itens.columns) else aplicar_extracao(itens)


class ComVeiculo:
    """Embrulha um recuperador (lexical, semantico ou hibrido) com a regra acima, para a AVALIACAO.

    O Buscador do app aplica a mesma funcao `reordenar` (com filtros), entao medir aqui e medir o que o
    app faz. Os scores devolvidos sao POSICOES invertidas (maior = melhor), nao cossenos: servem so para
    ranquear; zero continua significando "sem evidencia".
    """

    corpus_dependente = True

    def __init__(self, base, profundidade: int = PROFUNDIDADE) -> None:
        self.base = base
        self.profundidade = profundidade
        self.nome = f"{base.nome}+veiculo"
        self._itens: pd.DataFrame | None = None

    def fit_itens(self, itens: pd.DataFrame) -> None:
        """Ajusta a base com os textos E guarda os campos de veiculo e ano dos mesmos itens."""
        self._itens = garantir_extracao(itens).reset_index(drop=True)
        self.base.fit(self._itens["texto_busca"].tolist())

    def fit(self, textos) -> None:
        raise TypeError("ComVeiculo precisa dos itens (veículo e ano): use fit_itens(itens)")

    def score_rows(self, linhas):
        return self.base.score_rows(linhas)  # consulta = item do catalogo: sem veiculo a ordenar

    def score_texts(self, consultas) -> np.ndarray:
        if self._itens is None:
            raise RuntimeError("chame fit_itens antes de score_texts")
        S = np.asarray(self.base.score_texts(consultas), dtype=np.float64)
        saida = np.zeros_like(S)
        for i, texto in enumerate(consultas):
            r = np.round(S[i], DECIMAIS_EMPATE)
            ordem = np.argsort(-r, kind="stable")
            ordem = ordem[r[ordem] > 0]
            ordem = reordenar(ordem, self._itens, analisar_consulta(texto), self.profundidade)
            saida[i, ordem] = np.arange(len(ordem), 0, -1, dtype=np.float64)
        return saida


def normalizar_consulta(texto: str) -> str:
    """Mesma regra do texto de busca (decisao 21); usado so em testes e scripts."""
    from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA
    return normalizar_texto(texto, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA)
