"""Modo Adaptacoes (Etapa 4): "isso serve no meu carro?" com a honestidade que o catalogo permite. Motor, sem Streamlit.

Entrada: a peca (texto), o veiculo e o ano (campos; podem vir da extracao da consulta). Saida: blocos de
itens, cada um com um RÓTULO fechado e o MOTIVO de cada linha. Nunca ha "100% compativel":

- ROTULO_DECLARADO  a descricao do fornecedor cita o veiculo (e o ano cabe na faixa, se houver). E a
                    declaracao do fornecedor, nao uma garantia: a tela pede para confirmar antes de vender;
- ROTULO_SUGESTAO   tudo o mais ("sugestao a validar"): veiculo da mesma familia, peca generica sem
                    veiculo, mesmo GTIN em outro fornecedor, peca relacionada.

"Familia" = o que o PROPRIO catalogo agrupa (modelos citados juntos nas mesmas descricoes; ver
`veiculos.vizinhos_no_catalogo`). Nao ha conhecimento externo de plataformas: a camada de referencia de
compatibilidade (ACES/PIES) continua adiada. NCM so serve como categoria quando NAO e o generico 87089990.

Sem LLM e sem Ollama: tudo e determinístico. O candidato vem do `Buscador` (usa o modo hibrido se o Ollama
estiver no ar) pela descricao da PECA, sem o veiculo, para o nome do carro nao dominar a busca.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from autopecas.data.quality import MANTER_SIMBOLOS_NO_TEXTO_BUSCA, normalizar_texto
from autopecas.retrieval.veiculo import COBERTURA_MINIMA, cobertura, palavras_da_peca
from autopecas.search import Buscador, Filtros
from autopecas.veiculos import analisar_consulta, vizinhos_no_catalogo

ROTULO_DECLARADO = "citado na descrição do fornecedor"
ROTULO_SUGESTAO = "sugestão a validar"
ROTULOS_VALIDOS = frozenset({ROTULO_DECLARADO, ROTULO_SUGESTAO})
AVISO = (
    "Confirme a aplicação no veículo antes de vender: o catálogo não garante compatibilidade. "
    "Só aparece como “citado” o que a descrição do próprio fornecedor diz; o resto é sugestão a validar."
)
NCM_GENERICO = "87089990"
CANDIDATOS = 120          # itens buscados na base para classificar
MAX_POR_BLOCO = 15
MAX_RELACIONADAS = 8

COLUNAS_SAIDA = ["fornecedor", "codigo_interno", "descricao", "preco", "ncm", "gtin", "rotulo", "motivo"]


@dataclass(frozen=True)
class Bloco:
    titulo: str
    rotulo: str
    itens: pd.DataFrame  # COLUNAS_SAIDA

    def __post_init__(self) -> None:
        if self.rotulo not in ROTULOS_VALIDOS:
            raise ValueError(f"rótulo inválido: {self.rotulo!r}")


@dataclass(frozen=True)
class Adaptacoes:
    peca: str
    veiculo: str | None
    ano: int | None
    blocos: list[Bloco] = field(default_factory=list)
    fora_do_ano: int = 0       # itens que citam o veiculo mas com faixa de ano que nao cobre o pedido
    aviso: str = AVISO

    def total(self) -> int:
        return sum(len(b.itens) for b in self.blocos)


def _ncm_util(ncm) -> str | None:
    n = "" if pd.isna(ncm) else str(ncm).strip()
    return n if n and n != NCM_GENERICO else None


def _faixa_texto(ini, fim) -> str:
    if pd.isna(ini) and pd.isna(fim):
        return ""
    if pd.isna(fim):
        return f"a partir de {int(ini)}"
    if pd.isna(ini):
        return f"até {int(fim)}"
    return f"{int(ini)}–{int(fim)}"


def _cobre(ano, ini, fim) -> bool | None:
    """True/False se a faixa cobre o ano; None se o item nao tem faixa (ou nao ha ano pedido)."""
    if ano is None or (pd.isna(ini) and pd.isna(fim)):
        return None
    return bool((pd.isna(ini) or ini <= ano) and (pd.isna(fim) or ano <= fim))


def _nota_ano(ano, cobre, faixa: str) -> str:
    if ano is None:
        return f" · faixa {faixa} (sem ano no pedido)" if faixa else ""
    if cobre:
        return f" · faixa {faixa} cobre {ano}"
    return " · descrição sem ano: confirme o ano" if not faixa else ""


def _linhas(itens: pd.DataFrame, motivos: list[str], rotulo: str) -> pd.DataFrame:
    out = itens[["fornecedor", "codigo_interno", "descricao", "preco", "ncm", "gtin"]].copy()
    out["rotulo"] = rotulo
    out["motivo"] = motivos
    return out.reset_index(drop=True)


def sugerir(
    buscador: Buscador,
    peca: str,
    veiculo: str | None = None,
    ano: int | None = None,
    filtros: Filtros | None = None,
) -> Adaptacoes:
    """Monta os blocos de adaptacao. `veiculo` em minusculas, como no vocabulario (ex.: 'uno')."""
    veiculo = (veiculo or "").strip().lower() or None
    # a consulta ja pode trazer veiculo/ano misturados ("cabo capo uno 88"): campos explicitos valem mais
    lida = analisar_consulta(peca)
    if veiculo is None and lida.veiculos:
        veiculo = lida.veiculos[0]
    if ano is None:
        ano = lida.ano
    texto_peca = lida.peca if lida.tem_veiculo else normalizar_texto(
        peca, manter_simbolos=MANTER_SIMBOLOS_NO_TEXTO_BUSCA)
    palavras = palavras_da_peca(texto_peca)
    if not veiculo or not palavras:
        return Adaptacoes(peca, veiculo, ano)  # sem veiculo ou sem peca nao ha o que adaptar

    res = buscador.buscar(texto_peca, k=CANDIDATOS, filtros=filtros, por_veiculo=False)
    todos = res.itens
    textos_ok = [cobertura(palavras, t) > COBERTURA_MINIMA for t in todos["texto_busca"]]
    cand = todos[textos_ok].reset_index(drop=True)
    if cand.empty:
        return Adaptacoes(peca, veiculo, ano)

    viz = vizinhos_no_catalogo(buscador.itens).get(veiculo, {})
    citados, familia, genericos = [], [], []
    fora = 0
    for i, row in cand.iterrows():
        vs = list(row["veiculos"])
        cobre = _cobre(ano, row["ano_ini"], row["ano_fim"])
        faixa = _faixa_texto(row["ano_ini"], row["ano_fim"])
        if veiculo in vs:
            if cobre is False:
                fora += 1
                continue
            citados.append((i, f"cita {veiculo}" + _nota_ano(ano, cobre, faixa)))
        elif any(v in viz for v in vs):
            v = next(v for v in vs if v in viz)
            if cobre is False:
                fora += 1
                continue
            familia.append((i, f"cita {v}: o catálogo cita {veiculo} e {v} juntos em {viz[v]} descrições"
                            + _nota_ano(ano, cobre, faixa)))
        elif not vs:
            genericos.append((i, "descrição não cita veículo"))
        # cita so outro veiculo sem relacao no catalogo: nao e sugestao

    # categoria por NCM: so o NCM dos itens CITADOS, e so se nao for o generico 87089990
    ncms = {n for n in (_ncm_util(cand.at[i, "ncm"]) for i, _ in citados) if n}
    for k, (i, motivo) in enumerate(genericos):
        n = _ncm_util(cand.at[i, "ncm"])
        if n and n in ncms:
            genericos[k] = (i, f"{motivo} · mesmo NCM {n} dos itens citados")

    relacionadas = []
    if ncms:  # peca relacionada: mesmo NCM util dos citados, mas descricao de outra peca (reprovada na cobertura)
        for i, row in todos[[not ok for ok in textos_ok]].iterrows():
            n = _ncm_util(row["ncm"])
            if n in ncms and len(relacionadas) < MAX_RELACIONADAS:
                relacionadas.append((i, f"relacionada, não igual: mesmo NCM {n} dos itens citados, outra descrição"))

    def montar(titulo, rotulo, lista, fonte, limite=MAX_POR_BLOCO):
        lista = lista[:limite]
        if not lista:
            return None
        return Bloco(titulo, rotulo, _linhas(fonte.loc[[i for i, _ in lista]], [m for _, m in lista], rotulo))

    blocos = [
        montar("Citado na descrição do fornecedor (confira se é a mesma peça)", ROTULO_DECLARADO, citados, cand),
        montar("Sugestão a validar: veículo da mesma família no catálogo", ROTULO_SUGESTAO, familia, cand),
        montar("Sugestão a validar: mesma peça, descrição sem veículo", ROTULO_SUGESTAO, genericos, cand),
    ]
    if relacionadas:
        blocos.append(montar("Sugestão a validar: peças relacionadas (não são a mesma peça)", ROTULO_SUGESTAO,
                             relacionadas, todos, MAX_RELACIONADAS))
    bloco_gtin = _mesmo_gtin(buscador.itens, cand, [i for i, _ in citados])
    if bloco_gtin is not None:
        blocos.append(bloco_gtin)
    return Adaptacoes(peca, veiculo, ano, [b for b in blocos if b is not None], fora)


def _mesmo_gtin(itens: pd.DataFrame, cand: pd.DataFrame, posicoes_citados: list[int]) -> Bloco | None:
    """Itens de OUTRO fornecedor com o mesmo GTIN valido de um item citado (mesmo produto, aplicacao a validar)."""
    if not posicoes_citados:
        return None
    ja = set(zip(cand["fornecedor"], cand["codigo_interno"]))
    achados, motivos = [], []
    for i in posicoes_citados:
        g, forn = cand.at[i, "gtin_chave"], cand.at[i, "fornecedor"]
        if pd.isna(g):
            continue
        for _, outro in itens[(itens["gtin_chave"] == g) & (itens["fornecedor"] != forn)].iterrows():
            chave = (outro["fornecedor"], outro["codigo_interno"])
            if chave in ja:
                continue
            ja.add(chave)
            achados.append(outro)
            motivos.append(f"mesmo GTIN {g} do item {forn}:{cand.at[i, 'codigo_interno']}, que cita o veículo")
    if not achados:
        return None
    df = pd.DataFrame(achados)
    return Bloco("Sugestão a validar: mesmo produto em outro fornecedor (mesmo GTIN)", ROTULO_SUGESTAO,
                 _linhas(df, motivos, ROTULO_SUGESTAO))
