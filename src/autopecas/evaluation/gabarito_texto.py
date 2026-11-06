"""Gabarito manual em texto livre (perguntas escritas pelo grupo + pecas esperadas).

Serve onde nao ha gabarito por GTIN (nivel Basico) e para medir o comportamento com texto livre:
a consulta e uma pergunta, nao um item do catalogo.

Formato (CSV UTF-8, delimitador ';' ou ',' ou tabulacao; uma linha por pergunta):

    id;pergunta;pecas;tipo;autor;observacao
    Q01;cabo do capo do gol;C:02864;direta;Anderson;
    Q02;embreagem do fiesta;A:130781|C:07624;coloquial;Marcelo;duas pecas servem
    Q03;pastilha de freio de moto;;sem_resposta;Victor;nao existe no Basico

- `pecas`: FORNECEDOR:codigo_interno, separadas por '|' (ou ';' se a celula estiver entre aspas).
  O prefixo com letra impede o Excel de comer zeros a esquerda ('07624'). Todas as pecas que
  atendem a pergunta devem estar listadas (relevancia binaria).
- `tipo` (opcional): direta, coloquial, erro_digitacao, sem_veiculo, por_codigo ou sem_resposta.
  `sem_resposta` = pergunta cuja peca NAO existe no nivel; `pecas` fica vazio e ela nao entra
  nas metricas de acerto (so na revisao manual).
- `autor` e `observacao`: livres.

A leitura (`ler_gabarito`) valida so o arquivo; a conferencia contra o catalogo e feita por
`validar_contra_catalogo`, e `resolver` converte as pecas em posicoes do catalogo do nivel.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from autopecas.config import NIVEIS
from autopecas.data.levels import fornecedores_do_nivel, mascara_nivel

TIPO_SEM_RESPOSTA = "sem_resposta"
TIPO_NAO_INFORMADO = "nao_informado"
TIPOS = ("direta", "coloquial", "erro_digitacao", "sem_veiculo", "por_codigo", TIPO_SEM_RESPOSTA)
MIN_PERGUNTAS_COM_RESPOSTA = 30  # criterio de aceite da Etapa 3

_OBRIGATORIAS = ("id", "pergunta", "pecas")
_OPCIONAIS = ("tipo", "autor", "observacao")
_PECA = re.compile(r"^([A-Za-z]+)\s*:\s*(\S.*)$")


class ErroGabarito(ValueError):
    """Gabarito invalido. `problemas` lista tudo o que foi encontrado (nao so o primeiro erro)."""

    def __init__(self, problemas: list[str]) -> None:
        self.problemas = list(problemas)
        super().__init__("\n".join(f"- {p}" for p in self.problemas))


@dataclass(frozen=True)
class Pergunta:
    id: str
    texto: str
    pecas: tuple[tuple[str, str], ...]  # (fornecedor, codigo_interno), na ordem escrita
    tipo: str
    autor: str
    observacao: str
    linha: int  # linha do arquivo, para mensagens de erro

    @property
    def sem_resposta(self) -> bool:
        return self.tipo == TIPO_SEM_RESPOSTA


@dataclass(frozen=True)
class GabaritoTexto:
    """Gabarito resolvido contra o catalogo de um nivel.

    `relevantes[i]` sao posicoes em `itens_do_nivel(df, nivel)` (vazio nas sem_resposta).
    """

    nivel: str
    ids: tuple[str, ...]
    perguntas: tuple[str, ...]
    tipos: tuple[str, ...]
    relevantes: tuple[np.ndarray, ...]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def com_resposta(self) -> np.ndarray:
        return np.array([len(r) > 0 for r in self.relevantes], dtype=bool)


# ---------------------------------------------------------------- leitura
def _chave_coluna(nome: str) -> str:
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", "_", s.strip().lower())


def _ler_texto(caminho: Path | str) -> str:
    dados = Path(caminho).read_bytes()
    try:
        return dados.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:  # "CSV (separado por virgulas)" do Excel em portugues grava em Windows-1252
        return dados.decode("cp1252")
    except UnicodeDecodeError:
        raise ErroGabarito(["Não consegui ler o arquivo: salve como CSV UTF-8."]) from None


def _separar_pecas(celula: str) -> list[str]:
    return [t.strip() for t in re.split(r"[|;\n]+", celula) if t.strip()]


def ler_gabarito(caminho: Path | str) -> list[Pergunta]:
    """Le e valida o arquivo (sem consultar o catalogo). Acumula todos os erros e levanta ErroGabarito."""
    texto = _ler_texto(caminho)
    primeira = next((l for l in texto.splitlines() if l.strip()), "")
    delimitador = max(";", ",", "\t", key=primeira.count)
    leitor = csv.reader(io.StringIO(texto, newline=""), delimiter=delimitador)
    registros = [(leitor.line_num, linha) for linha in leitor]
    registros = [(n, [c.strip() for c in linha]) for n, linha in registros if any(c.strip() for c in linha)]
    if not registros:
        raise ErroGabarito(["Arquivo vazio."])

    cabecalho = [_chave_coluna(c) for c in registros[0][1]]
    faltam = [c for c in _OBRIGATORIAS if c not in cabecalho]
    if faltam:
        raise ErroGabarito([
            f"Faltam colunas no cabeçalho: {', '.join(faltam)}. "
            f"Esperado: {', '.join(_OBRIGATORIAS + _OPCIONAIS)} (cabeçalho lido: {', '.join(cabecalho)})."
        ])

    erros: list[str] = []
    perguntas: list[Pergunta] = []
    ids_vistos: dict[str, int] = {}
    for n, linha in registros[1:]:
        campos = dict(zip(cabecalho, linha + [""] * (len(cabecalho) - len(linha))))
        problemas: list[str] = []
        id_ = campos["id"]
        if not id_:
            problemas.append("id vazio")
        elif id_ in ids_vistos:
            problemas.append(f"id '{id_}' repetido (já usado na linha {ids_vistos[id_]})")
        else:
            ids_vistos[id_] = n
        if not campos["pergunta"]:
            problemas.append("pergunta vazia")

        pecas: list[tuple[str, str]] = []
        for token in _separar_pecas(campos["pecas"]):
            m = _PECA.match(token)
            if not m:
                problemas.append(f"peça '{token}' fora do formato FORNECEDOR:código (ex.: C:02864)")
                continue
            peca = (m.group(1).upper(), m.group(2).strip())
            if peca in pecas:
                problemas.append(f"peça {peca[0]}:{peca[1]} repetida na mesma pergunta")
            else:
                pecas.append(peca)

        tipo = campos.get("tipo", "").lower()
        if tipo and tipo not in TIPOS:
            problemas.append(f"tipo '{tipo}' inválido (use: {', '.join(TIPOS)})")
        elif tipo == TIPO_SEM_RESPOSTA and pecas:
            problemas.append("tipo sem_resposta não pode ter peças esperadas")
        elif tipo != TIPO_SEM_RESPOSTA and not pecas and not any("fora do formato" in p for p in problemas):
            problemas.append("sem peças esperadas: preencha `pecas` ou use tipo=sem_resposta")
        if not tipo:
            tipo = TIPO_NAO_INFORMADO

        if problemas:
            erros.extend(f"linha {n} ({id_ or 'sem id'}): {p}" for p in problemas)
            continue
        perguntas.append(Pergunta(id_, campos["pergunta"], tuple(pecas), tipo,
                                  campos.get("autor", ""), campos.get("observacao", ""), n))
    if erros:
        raise ErroGabarito(erros)
    return perguntas


# ---------------------------------------------------------------- catalogo
def itens_do_nivel(df: pd.DataFrame, nivel: str) -> pd.DataFrame:
    """Itens do nivel, com indice 0..n-1: as posicoes do gabarito valem para ESTA tabela."""
    return df[mascara_nivel(df, nivel)].reset_index(drop=True)


def validar_contra_catalogo(perguntas: list[Pergunta], df: pd.DataFrame, nivel: str) -> list[str]:
    """Erros de conferencia com o catalogo do nivel (lista vazia = tudo certo)."""
    sub = itens_do_nivel(df, nivel)
    nivel_letras = fornecedores_do_nivel(df, nivel)
    codigos: dict[str, set[str]] = defaultdict(set)
    for f, c in zip(sub["fornecedor"], sub["codigo_interno"]):
        codigos[f].add(str(c).strip())

    erros: list[str] = []
    for p in perguntas:
        for forn, cod in p.pecas:
            onde = f"linha {p.linha} ({p.id}): peça {forn}:{cod}"
            if forn not in nivel_letras:
                erros.append(f"{onde} — o fornecedor {forn} não faz parte do nível {nivel} "
                             f"({', '.join(nivel_letras)})")
            elif cod not in codigos[forn]:
                parecidos = sorted(c for c in codigos[forn] if c.lstrip("0") == cod.lstrip("0"))
                dica = f" Você quis dizer {forn}:{parecidos[0]}? (o Excel remove zeros à esquerda)" if parecidos else ""
                erros.append(f"{onde} — código não existe no catálogo do fornecedor {forn}.{dica}")
    return erros


def avisos_do_gabarito(perguntas: list[Pergunta]) -> list[str]:
    n = sum(1 for p in perguntas if not p.sem_resposta)
    if n < MIN_PERGUNTAS_COM_RESPOSTA:
        return [f"Só {n} pergunta(s) com resposta; o critério da Etapa 3 pede pelo menos "
                f"{MIN_PERGUNTAS_COM_RESPOSTA}."]
    return []


def resumo_do_gabarito(perguntas: list[Pergunta]) -> dict:
    tipos: dict[str, int] = defaultdict(int)
    autores: dict[str, int] = defaultdict(int)
    for p in perguntas:
        tipos[p.tipo] += 1
        autores[p.autor or "(sem autor)"] += 1
    return {
        "perguntas": len(perguntas),
        "com_resposta": sum(1 for p in perguntas if not p.sem_resposta),
        "sem_resposta": sum(1 for p in perguntas if p.sem_resposta),
        "pecas_distintas": len({peca for p in perguntas for peca in p.pecas}),
        "por_tipo": dict(sorted(tipos.items())),
        "por_autor": dict(sorted(autores.items())),
    }


def resolver(perguntas: list[Pergunta], df: pd.DataFrame, nivel: str = "Básico") -> GabaritoTexto:
    """Converte as pecas em posicoes de `itens_do_nivel(df, nivel)`. Levanta ErroGabarito se nao bater."""
    if nivel not in NIVEIS:
        raise ValueError(f"nível desconhecido: {nivel}")
    erros = validar_contra_catalogo(perguntas, df, nivel)
    if erros:
        raise ErroGabarito(erros)
    sub = itens_do_nivel(df, nivel)
    posicoes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pos, (f, c) in enumerate(zip(sub["fornecedor"], sub["codigo_interno"])):
        posicoes[(f, str(c).strip())].append(pos)  # se o catalogo repete a linha, todas valem
    relevantes = tuple(
        np.array(sorted({q for peca in p.pecas for q in posicoes[peca]}), dtype=int) for p in perguntas
    )
    return GabaritoTexto(
        nivel=nivel,
        ids=tuple(p.id for p in perguntas),
        perguntas=tuple(p.texto for p in perguntas),
        tipos=tuple(p.tipo for p in perguntas),
        relevantes=relevantes,
    )
