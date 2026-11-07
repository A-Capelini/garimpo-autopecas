"""Sugestao de pecas esperadas para o holdout por FILTRO EXAUSTIVO de palavras-chave (Etapa 4, fase A).

Por que existe: as perguntas do holdout foram escritas por um modelo (Gemini) sem ver o catalogo, entao
nao trazem as pecas esperadas. Quem acha as pecas deve ser um filtro simples sobre a planilha, e NAO o
ranking do sistema (decisao 33): quem usa o sistema para achar a resposta mede o sistema contra ele mesmo.
Por isso este modulo NAO usa TF-IDF, embeddings nem a regra de ordenacao por veiculo. Ele so:

1. separa veiculo e ano da pergunta (`veiculos.analisar_consulta`);
2. separa os termos da PECA (nucleo) dos termos de lado/posicao (esq, dir, diant, tras, sup, inf, int, ext);
3. casa cada termo com as palavras do catalogo por igualdade, plural, abreviacao (`diant` = `dianteiro`,
   `interr` = `interruptor`) ou, so se nada casar, por similaridade de grafia (erros de digitacao);
4. lista TODOS os itens que casam com todos os termos conhecidos (e, com 3 ou mais termos, os que casam
   com todos menos um, marcados como parciais);
5. aplica o criterio de rotulagem do gabarito do Basico (docs/GABARITO_BASICO.md) como TRIAGEM:
   com veiculo na pergunta so o item que cita o veiculo e "provavel"; generico ou de outro veiculo e
   "improvavel"; lado/posicao conflitante e "improvavel".

A triagem e uma dica para quem confere. O gabarito so vale depois que o grupo marcar S/N em cada
candidata (decisoes 36 e 45). O criterio "peca parecida mas diferente nao entra" (coifa != junta
homocinetica, tampa != reservatorio) depende de leitura humana e nao e decidido aqui.

Limites conhecidos: motorista = lado esquerdo e carona = lado direito (direcao a esquerda, Brasil);
termo da pergunta sem correspondente no catalogo e ignorado no filtro e listado em `sem_correspondencia`
(pode ser sinonimo que so o grupo conhece); o ano so entra na triagem quando o item declara faixa de anos.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from autopecas.data.quality import normalizar_texto
from autopecas.veiculos import Extracao, analisar_consulta, extrair

MAX_CANDIDATAS = 40  # por pergunta; o total real do filtro fica em `Sugestao.total_filtro`
CUTOFF_GRAFIA = 0.78  # similaridade minima (difflib) para tratar um termo sem casamento como erro de digitacao

STOP = frozenset(
    "de do da dos das pro pra para por e o a os as um uma com sem no na nos nas em ao aos ou c p lado lados".split()
)
FRACAS = frozenset(  # descrevem a forma de venda ou um atributo e nao a peca; nao sao exigidas
    "jogo jogos kit kits par pares completo completa simples duplo dupla duplos novo nova original standard".split()
)

TRIAGEM_ORDEM = {"provavel": 0, "possivel": 1, "improvavel": 2}

_EIXOS: dict[str, dict[str, re.Pattern]] = {
    "lado": {
        "LE": re.compile(r"^(le|esq|esquerd[oa]s?|motorista)$"),  # motorista = esquerda (direcao a esquerda)
        "LD": re.compile(r"^(ld|dir|direit[oa]s?|carona)$"),
    },
    "posicao": {"DIANT": re.compile(r"^dian\w*$"), "TRAS": re.compile(r"^tras\w*$")},
    "altura": {"SUP": re.compile(r"^(sup|superior)$"), "INF": re.compile(r"^(inf|inferior)$")},
    "face": {"INT": re.compile(r"^(int|intern[oa]s?)$"), "EXT": re.compile(r"^(ext|extern[oa]s?)$")},
}
_ROTULO_EIXO = {"lado": "lado", "posicao": "posição", "altura": "altura", "face": "face"}


# ---------------------------------------------------------------- texto
def _tokens(texto: str) -> list[str]:
    return [t for t in normalizar_texto(texto, manter_simbolos=False).split() if t]


def _eixos(tokens) -> dict[str, frozenset[str]]:
    achados: dict[str, set[str]] = defaultdict(set)
    for t in tokens:
        for eixo, valores in _EIXOS.items():
            for valor, padrao in valores.items():
                if padrao.match(t):
                    achados[eixo].add(valor)
    return {e: frozenset(v) for e, v in achados.items()}


def _eh_eixo(token: str) -> bool:
    return bool(_eixos([token]))


def _raiz(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def casa(token: str, palavra: str) -> str | None:
    """'exato', 'abreviacao' ou None. `token` vem da pergunta; `palavra` do catalogo (ambos normalizados)."""
    if token == palavra:
        return "exato"
    if any(ch.isdigit() for ch in token + palavra):
        return None  # medidas e numeros (12v, 13mm) so casam por igualdade
    rt, rp = _raiz(token), _raiz(palavra)
    if rt == rp:
        return "exato"
    if palavra not in STOP and palavra not in FRACAS and len(rp) >= 3 and len(rt) - len(rp) >= 2 and rt.startswith(rp):
        return "abreviacao"  # o catalogo abrevia: ign -> ignicao, interr -> interruptor
    if len(rt) >= 4 and len(rp) - len(rt) >= 2 and rp.startswith(rt):
        return "abreviacao"  # a pergunta abrevia: diant -> dianteiro, tras -> traseiro
    return None


# ---------------------------------------------------------------- indice
@dataclass(frozen=True)
class Indice:
    nivel: str
    itens: pd.DataFrame  # fornecedor, codigo_interno, descricao (indice 0..n-1)
    invertido: dict[str, list[int]]
    vocabulario: tuple[str, ...]
    extracoes: tuple[Extracao, ...]
    eixos: tuple[dict, ...]


def montar_indice(itens: pd.DataFrame, nivel: str = "") -> Indice:
    """`itens`: so as linhas do nivel (use `gabarito_texto.itens_do_nivel`)."""
    itens = itens.reset_index(drop=True)
    invertido: dict[str, list[int]] = defaultdict(list)
    eixos = []
    for i, desc in enumerate(itens["descricao"]):
        toks = _tokens(desc)
        for t in set(toks):
            invertido[t].append(i)
        eixos.append(_eixos(toks))
    extracoes = tuple(extrair(d) for d in itens["descricao"])
    return Indice(nivel, itens, dict(invertido), tuple(sorted(invertido)), extracoes, tuple(eixos))


# ---------------------------------------------------------------- resultado
@dataclass(frozen=True)
class Candidata:
    posicao: int  # linha em `Indice.itens`
    fornecedor: str
    codigo: str
    descricao: str
    camada: str  # 'completa' (casa todos os termos) ou 'parcial' (todos menos um)
    casamentos: str  # 'bomba=bomba; agua=agua' (termo=palavra do catalogo; ~ = por similaridade de grafia)
    veiculo: str  # cita_o_veiculo | generica | outro_veiculo | sem_veiculo_na_pergunta
    conflitos: str  # eixos (lado, posição...) em que o item contraria a pergunta; vazio = nenhum
    triagem: str  # provavel | possivel | improvavel
    motivo: str


@dataclass(frozen=True)
class Sugestao:
    id: str
    pergunta: str
    veiculos: tuple[str, ...]
    ano: int | None
    termos: tuple[str, ...]  # nucleo exigido
    sem_correspondencia: tuple[str, ...]  # termos da pergunta que nada no catalogo casa (ignorados no filtro)
    total_filtro: int  # itens que passaram no filtro (completos + parciais), antes do corte
    candidatas: tuple[Candidata, ...]  # ate MAX_CANDIDATAS, mais promissoras primeiro
    contagem: dict  # triagem -> quantidade, sobre TODOS os itens do filtro

    @property
    def existe_peca(self) -> str:
        """'sim' (ha provavel), 'talvez' (so possiveis), 'so_improvavel' (so itens que o criterio descarta,
        ex.: genericos quando a pergunta cita veiculo) ou 'nada' (nenhum item passou no filtro)."""
        if self.contagem.get("provavel", 0):
            return "sim"
        if self.contagem.get("possivel", 0):
            return "talvez"
        return "so_improvavel" if self.contagem.get("improvavel", 0) else "nada"


def _expandir(token: str, indice: Indice) -> dict[str, str]:
    """palavra do catalogo -> tipo de casamento ('exato', 'abreviacao' ou 'grafia')."""
    achadas = {}
    for palavra in indice.vocabulario:
        tipo = casa(token, palavra)
        if tipo:
            achadas[palavra] = tipo
    if not achadas and len(token) >= 5 and not any(ch.isdigit() for ch in token):
        for palavra in difflib.get_close_matches(token, indice.vocabulario, n=3, cutoff=CUTOFF_GRAFIA):
            achadas[palavra] = "grafia"
    return achadas


def _triar(camada, veiculo, ano_status, conflitos, grafia, sem_correspondencia) -> tuple[str, str]:
    if conflitos:
        return "improvavel", f"contraria a pergunta em: {conflitos}"
    if veiculo == "outro_veiculo":
        return "improvavel", "cita só outro veículo"
    if veiculo == "generica":
        return "improvavel", "genérica (sem veículo); pelo critério não entra quando a pergunta cita veículo"
    if ano_status == "fora":
        return "improvavel", "faixa de anos do item não cobre o ano da pergunta"
    motivos = []
    if camada == "parcial":
        motivos.append("casa só parte dos termos")
    if grafia:
        motivos.append("casou por similaridade de grafia")
    if sem_correspondencia:
        motivos.append("há termo da pergunta sem correspondente no catálogo")
    if motivos:
        return "possivel", "; ".join(motivos)
    return "provavel", "casa todos os termos" + ("; cita o veículo" if veiculo == "cita_o_veiculo" else "")


def sugerir(id_: str, pergunta: str, indice: Indice, max_candidatas: int = MAX_CANDIDATAS) -> Sugestao:
    consulta = analisar_consulta(pergunta)
    tokens = [t for t in _tokens(consulta.peca) if len(t) > 1 and t not in STOP]
    eixos_pergunta = _eixos(tokens)
    nucleo = list(dict.fromkeys(t for t in tokens if t not in FRACAS and not _eh_eixo(t)))

    casamentos: dict[str, dict[str, str]] = {}
    sem_correspondencia: list[str] = []
    for t in nucleo:
        achadas = _expandir(t, indice)
        if achadas:
            casamentos[t] = achadas
        else:
            sem_correspondencia.append(t)
    conhecidos = list(casamentos)

    vazia = Sugestao(id_, pergunta, consulta.veiculos, consulta.ano, tuple(conhecidos),
                     tuple(sem_correspondencia), 0, (), {})
    if not conhecidos:
        return vazia

    por_termo = {t: {i for w in casamentos[t] for i in indice.invertido[w]} for t in conhecidos}
    contagem_termos = Counter(i for conjunto in por_termo.values() for i in conjunto)
    n = len(conhecidos)
    camada_do_item = {i: "completa" for i, k in contagem_termos.items() if k == n}
    if n >= 3:
        camada_do_item.update({i: "parcial" for i, k in contagem_termos.items() if k == n - 1})

    candidatas = []
    for i, camada in camada_do_item.items():
        itens_i = indice.itens.iloc[i]
        ext = indice.extracoes[i]
        if not consulta.veiculos:
            veiculo = "sem_veiculo_na_pergunta"
        elif set(ext.veiculos) & set(consulta.veiculos):
            veiculo = "cita_o_veiculo"
        else:
            veiculo = "outro_veiculo" if ext.veiculos else "generica"
        ano_status = "n/d"
        if veiculo == "cita_o_veiculo" and consulta.ano is not None and ext.tem_ano:
            ano_status = "dentro" if ext.cobre(consulta.ano) else "fora"
        conflitos = [
            _ROTULO_EIXO[eixo]
            for eixo, valores in eixos_pergunta.items()
            if indice.eixos[i].get(eixo) and not (valores & indice.eixos[i][eixo])
        ]
        usados = []
        grafia = False
        for t in conhecidos:
            palavras = [(w, tipo) for w, tipo in casamentos[t].items() if i in indice.invertido[w]]
            if palavras:
                w, tipo = sorted(palavras, key=lambda p: (p[1] != "exato", p[0]))[0]
                grafia |= tipo == "grafia"
                usados.append(f"{t}{'~' if tipo == 'grafia' else '='}{w}")
        triagem, motivo = _triar(camada, veiculo, ano_status, ", ".join(conflitos), grafia, sem_correspondencia)
        candidatas.append(Candidata(
            i, str(itens_i["fornecedor"]), str(itens_i["codigo_interno"]).strip(), str(itens_i["descricao"]),
            camada, "; ".join(usados), veiculo, ", ".join(conflitos), triagem, motivo))

    contagem = dict(Counter(c.triagem for c in candidatas))
    candidatas.sort(key=lambda c: (TRIAGEM_ORDEM[c.triagem], c.camada != "completa", c.fornecedor, c.codigo))
    return Sugestao(id_, pergunta, consulta.veiculos, consulta.ano, tuple(conhecidos), tuple(sem_correspondencia),
                    len(candidatas), tuple(candidatas[:max_candidatas]), contagem)


# ---------------------------------------------------------------- fechamento (planilha revisada -> gabarito)
COLUNAS_GABARITO = ["id", "pergunta", "pecas", "tipo", "autor", "observacao"]
VALORES_SERVE = ("S", "N")


def gabarito_de_conferencia(
    perguntas: pd.DataFrame, revisadas: pd.DataFrame, extras: dict[str, str] | None = None,
    fornecedores: frozenset[str] | set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Monta o gabarito do holdout a partir da conferencia do grupo.

    perguntas: id, pergunta, tipo, autor, observacao (o CSV original do holdout).
    revisadas: id, fornecedor, codigo, serve (S/N/?), revisor (uma linha por candidata).
    extras: id -> 'FORN:codigo|FORN:codigo' de pecas que o filtro nao achou e o grupo acrescentou.

    Regras: pergunta sem nenhuma peca marcada S vira `sem_resposta` (a peca nao existe no nivel); pergunta
    que era `sem_resposta` e passou a ter peca fica sem tipo (o grupo escolhe) e a observacao conta isso.
    fornecedores: se dado, so conta o nivel que ele define (ex.: A a D para o Basico): candidatas e extras de
    outros fornecedores sao ignoradas, inclusive na checagem de pendencias. Assim uma conferencia feita no
    Intermediario gera tambem o gabarito do Basico.
    Retorna (gabarito, pendencias). Com pendencias, o gabarito ainda NAO deve ser usado.
    """
    extras = extras or {}
    pendencias: list[str] = []
    marcadas: dict[str, list[tuple[str, str]]] = defaultdict(list)
    revisores: dict[str, set[str]] = defaultdict(set)
    ids_validos = set(perguntas["id"])
    for _, r in revisadas.iterrows():
        id_ = str(r["id"]).strip()
        serve = str(r.get("serve", "") or "").strip().upper()
        cod = f"{r['fornecedor']}:{r['codigo']}"
        if fornecedores is not None and str(r["fornecedor"]).strip() not in fornecedores:
            continue
        revisor = str(r.get("revisor", "") or "").strip()
        if id_ not in ids_validos:
            pendencias.append(f"{id_}: id não existe no holdout (linha da candidata {cod})")
            continue
        if serve not in VALORES_SERVE:
            pendencias.append(f"{id_}: candidata {cod} sem decisão (use S ou N; '?' ou vazio não vale)")
            continue
        if not revisor:
            pendencias.append(f"{id_}: candidata {cod} sem o nome de quem revisou")
        else:
            revisores[id_].add(revisor)
        if serve == "S":
            marcadas[id_].append((str(r["fornecedor"]), str(r["codigo"])))

    linhas = []
    for _, p in perguntas.iterrows():
        id_ = p["id"]
        pecas = list(marcadas.get(id_, []))
        for token in re.split(r"[|;\n]+", extras.get(id_, "") or ""):
            token = token.strip()
            if not token:
                continue
            forn, _, cod = token.partition(":")
            if not cod:
                pendencias.append(f"{id_}: peça extra '{token}' fora do formato FORNECEDOR:código")
            elif fornecedores is not None and forn.strip().upper() not in fornecedores:
                continue
            elif (forn.strip().upper(), cod.strip()) not in pecas:
                pecas.append((forn.strip().upper(), cod.strip()))
        tipo = str(p.get("tipo", "") or "")
        obs = [str(p.get("observacao", "") or "").strip()]
        if not pecas:
            if tipo != "sem_resposta":
                obs.append(f"tipo original {tipo}; sem peça no nível após a conferência")
            tipo = "sem_resposta"
        elif tipo == "sem_resposta":
            obs.append("tipo original sem_resposta (Gemini); a conferência achou peça: escolher o tipo")
            tipo = ""
        if revisores.get(id_):
            obs.append("revisado por " + ", ".join(sorted(revisores[id_])))
        linhas.append({
            "id": id_, "pergunta": p["pergunta"], "pecas": "|".join(f"{f}:{c}" for f, c in pecas),
            "tipo": tipo, "autor": p.get("autor", ""), "observacao": " | ".join(o for o in obs if o),
        })
    return pd.DataFrame(linhas, columns=COLUNAS_GABARITO), pendencias
