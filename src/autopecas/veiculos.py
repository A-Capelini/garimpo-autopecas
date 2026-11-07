"""Extracao de veiculo e ano a partir de texto livre (descricao do catalogo ou consulta). Motor, sem Streamlit.

Decisoes da Etapa 4 (CONTEXTO, secao 6):
- os campos sao derivados EM MEMORIA (`aplicar_extracao`); o esquema do SQLite e o `texto_busca` nao
  mudam, entao os hashes e os vetores ja calculados continuam valendo;
- o vocabulario abaixo vem de evidencia do proprio catalogo (nomes completos e siglas confirmadas
  pelas listas `GOL/VOY/PAR/SAV`, `FUSCA/BRAS/VAR`); nao ha conhecimento externo de plataformas
  (a camada ACES/PIES continua adiada);
- siglas ambiguas ficam de fora: `PST`, `BRAS`, `PAR` solto (= "par" de pecas), `FIRE` (motor), `TIPO`.

Limites conhecidos (v1): o ano e uma faixa por item (envelope de todas as faixas citadas), sem
associar cada faixa a um veiculo; marcas (VW, FIAT...) nao contam como veiculo; erro de digitacao no
nome do veiculo nao e corrigido.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from autopecas.data.quality import normalizar_texto

# canonico -> aliases (alem do proprio nome), todos em minusculas sem acento.
VEICULOS: dict[str, tuple[str, ...]] = {
    "gol": (), "parati": ("parat",), "saveiro": ("sav",), "voyage": ("voy",),
    "santana": (), "passat": (), "polo": (), "fox": (), "golf": (),
    "fusca": ("fuscao",), "kombi": (), "brasilia": (), "variant": (),
    "uno": (), "mille": (), "premio": (), "elba": (), "fiorino": (),
    "palio": (), "siena": (), "strada": (),
    "corsa": (), "celta": (), "prisma": (), "monza": (), "kadett": (), "ipanema": (),
    "chevette": (), "opala": (), "caravan": ("caravana",), "vectra": (), "astra": (),
    "omega": (), "meriva": (), "zafira": (), "montana": (),
    "escort": (), "verona": (), "fiesta": (), "ka": (), "corcel": (), "belina": (),
    "del rey": ("delrey",), "pampa": (),
}

_ALIAS_PARA_CANONICO: dict[str, str] = {}
for _canon, _aliases in VEICULOS.items():
    for _nome in (_canon, *_aliases):
        _ALIAS_PARA_CANONICO[_nome] = _canon
_PADRAO_VEICULO = re.compile(
    r"(?<![a-z0-9])(" + "|".join(sorted(map(re.escape, _ALIAS_PARA_CANONICO), key=len, reverse=True)) + r")(?![a-z0-9])"
)
# "PAR" so e Parati dentro de lista com barra (GOL/VOY/PAR/SAV); "(PAR)" e "C/2 PAR" sao "par" de pecas.
_PAR_EM_LISTA = re.compile(r"(?:/par(?![a-z0-9])|(?<![a-z0-9])par/)")
_SEDAN = re.compile(r"(?<![a-z0-9])sedan(?![a-z0-9])")

PIVO_SECULO = 30  # ano de 2 digitos <= 30 vira 20xx; acima, 19xx (catalogo de carros antigos)
ANO_MIN, ANO_MAX = 1950, 2035

_N = r"(\d{4}|\d{2})"
_FAIXA = re.compile(rf"(?<![\d.,x]){_N}\s*/\s*{_N}(?![\d.,x])")
_FAIXA_A = re.compile(rf"(?<![\d.,x/]){_N}\s+a\s+{_N}(?![\d.,x/])")  # "78 A 82"; "08/99 A 08/03" (mes/ano) nao casa
_INICIO_ABERTO = re.compile(rf"(?<![\d.,x/]){_N}\s*/(?!\s*[\d\w])")
_FIM_ABERTO = re.compile(rf"(?<![\w/])/\s*{_N}(?![\d.,x/])")
_A_PARTIR = re.compile(rf"(?<![\d.,x/]){_N}\s*>")
_ATE = re.compile(rf"(?<![a-z0-9])ate\s*{_N}(?![\d.,x/])")
_ANO_4D = re.compile(r"(?<![\d.,x/])(19[5-9]\d|20[0-3]\d)(?![\d.,x/])")

NIVEL_DECLARADO = "declarado no catálogo"
NIVEL_SUGESTAO = "sugestão a validar"


def _ano(token: str) -> int | None:
    """'85' -> 1985, '09' -> 2009, '1998' -> 1998; fora de 1950..2035 -> None (cilindrada, medida...)."""
    n = int(token)
    if len(token) == 2:
        n += 2000 if n <= PIVO_SECULO else 1900
    return n if ANO_MIN <= n <= ANO_MAX else None


@dataclass(frozen=True)
class Extracao:
    veiculos: tuple[str, ...] = ()
    ano_ini: int | None = None
    ano_fim: int | None = None  # None com ano_ini preenchido = faixa aberta ("74/", "92>")

    @property
    def tem_ano(self) -> bool:
        return self.ano_ini is not None or self.ano_fim is not None

    def cobre(self, ano: int) -> bool:
        """True se `ano` cai na faixa (extremos abertos valem para qualquer ano do lado aberto)."""
        if not self.tem_ano:
            return False
        return (self.ano_ini is None or self.ano_ini <= ano) and (self.ano_fim is None or ano <= self.ano_fim)


def _texto(descricao) -> str:
    return normalizar_texto(descricao, manter_simbolos=True)  # mantem '/' e '>' (faixas de ano)


def extrair_veiculos(descricao) -> tuple[str, ...]:
    t = _texto(descricao)
    achados = [_ALIAS_PARA_CANONICO[m.group(1)] for m in _PADRAO_VEICULO.finditer(t)]
    if _PAR_EM_LISTA.search(t):
        achados.append("parati")
    if _SEDAN.search(t) and (not achados or "vw sedan" in t):  # "CORSA SEDAN" / "VECTRA SEDAN" nao sao Fusca
        achados.append("fusca")
    return tuple(dict.fromkeys(achados))


def extrair_anos(descricao) -> tuple[int | None, int | None]:
    t = _texto(descricao)
    inis: list[int] = []
    fins: list[int] = []
    aberto_fim = False
    consumidos: list[tuple[int, int]] = []

    def livre(m) -> bool:
        return not any(a < m.end() and m.start() < b for a, b in consumidos)

    for padrao in (_FAIXA, _FAIXA_A):
        for m in padrao.finditer(t):
            a, b = _ano(m.group(1)), _ano(m.group(2))
            if a is not None and b is not None and a <= b and livre(m):
                inis.append(a), fins.append(b), consumidos.append(m.span())
    for m in _INICIO_ABERTO.finditer(t):
        a = _ano(m.group(1))
        if a is not None and livre(m):
            inis.append(a), consumidos.append(m.span())
            aberto_fim = True
    for m in _A_PARTIR.finditer(t):
        a = _ano(m.group(1))
        if a is not None and livre(m):
            inis.append(a), consumidos.append(m.span())
            aberto_fim = True
    for m in _FIM_ABERTO.finditer(t):
        b = _ano(m.group(1))
        if b is not None and livre(m):
            fins.append(b), consumidos.append(m.span())
    for m in _ATE.finditer(t):
        b = _ano(m.group(1))
        if b is not None and livre(m):
            fins.append(b), consumidos.append(m.span())
    if not inis and not fins:
        for m in _ANO_4D.finditer(t):
            if livre(m):
                inis.append(int(m.group(1))), fins.append(int(m.group(1)))
    ini = min(inis) if inis else None
    fim = None if aberto_fim else (max(fins) if fins else None)
    return ini, fim


def extrair(descricao) -> Extracao:
    ini, fim = extrair_anos(descricao)
    return Extracao(extrair_veiculos(descricao), ini, fim)


def aplicar_extracao(df: pd.DataFrame, coluna: str = "descricao") -> pd.DataFrame:
    """Copia de `df` com `veiculos` (tupla), `ano_ini` e `ano_fim` (Int64; vazio = sem informacao)."""
    out = df.copy()
    ext = [extrair(d) for d in out[coluna]]
    out["veiculos"] = [e.veiculos for e in ext]
    out["ano_ini"] = pd.array([e.ano_ini for e in ext], dtype="Int64")
    out["ano_fim"] = pd.array([e.ano_fim for e in ext], dtype="Int64")
    return out


# ---------------------------------------------------------------- consulta
@dataclass(frozen=True)
class ConsultaAnalisada:
    original: str
    peca: str  # consulta normalizada SEM o veiculo e o ano (o que descreve a peca)
    veiculos: tuple[str, ...]
    ano: int | None

    @property
    def tem_veiculo(self) -> bool:
        return bool(self.veiculos)


_ANO_NA_CONSULTA = re.compile(r"(?<![\d.,x/])(19[5-9]\d|20[0-3]\d|\d{2})(?![\d.,x/])")


def analisar_consulta(consulta: str) -> ConsultaAnalisada:
    """Separa veiculo e ano do texto da pessoa. Ex.: 'cabo do capo do uno 88' -> peca 'cabo do capo do', uno, 1988.

    O ano so e lido quando ha veiculo na consulta (numero solto como '24' em 'palheta de 24 polegadas'
    e medida, nao ano) e quando e um ano plausivel (4 digitos, ou 2 digitos que formem 1950..2035).
    """
    veiculos = extrair_veiculos(consulta)
    t = _texto(consulta)
    ano = None
    if veiculos:
        for m in _ANO_NA_CONSULTA.finditer(t):
            a = _ano(m.group(1))
            if a is not None and (len(m.group(1)) == 4 or not _seguido_de_unidade(t, m.end())):
                ano = a
                t = t[: m.start()] + " " + t[m.end():]
                break
    for m in sorted(_PADRAO_VEICULO.finditer(t), key=lambda m: -m.start()):
        t = t[: m.start()] + " " + t[m.end():]
    t = _PAR_EM_LISTA.sub(" ", t)
    return ConsultaAnalisada(consulta, re.sub(r"\s+", " ", re.sub(r"[/>]", " ", t)).strip(), veiculos, ano)


_UNIDADES = ("pol", "polegada", "polegadas", "mm", "cm", "v", "volts", "amp", "w", "l", "litros", "kg")


def _seguido_de_unidade(t: str, fim: int) -> bool:
    resto = t[fim:].split()
    return bool(resto) and resto[0] in _UNIDADES


# ---------------------------------------------------------------- vizinhanca ("familia" pelo catalogo)
MIN_PARES = 5      # modelos citados juntos em pelo menos 5 descricoes ...
MIN_RAZAO = 0.30   # ... e em pelo menos 30% das descricoes do modelo MENOS citado


def vizinhos_no_catalogo(df: pd.DataFrame, coluna: str = "veiculos") -> dict[str, dict[str, int]]:
    """Modelo -> {outro modelo: n de descricoes em que o catalogo cita os dois}, so pares fortes.

    "Familia" = o que o PROPRIO catalogo agrupa (ex.: 'GOL/PARATI/SAVEIRO'). Nao e compatibilidade
    confirmada: serve so para SUGERIR a validar. Nao transitivo (A~B e B~C nao implica A~C).
    """
    cont: Counter = Counter()
    pares: Counter = Counter()
    for vs in df[coluna]:
        cont.update(vs)
        for a, b in itertools.combinations(sorted(vs), 2):
            pares[(a, b)] += 1
    viz: dict[str, dict[str, int]] = {}
    for (a, b), n in pares.items():
        if n >= MIN_PARES and n / min(cont[a], cont[b]) >= MIN_RAZAO:
            viz.setdefault(a, {})[b] = n
            viz.setdefault(b, {})[a] = n
    return viz
