"""Explicacao por LLM local (Etapa 5, decisoes 51 a 58). Motor, sem Streamlit.

Papel do LLM: resumir em ate 3 frases os 3 a 5 primeiros itens que a busca JA devolveu. A lista de
resultados nunca depende dele (decisao 2): qualquer falha aqui vira uma mensagem curta e a lista continua.

Principios (decisoes 56 a 58):
- o rotulo "sugestao a validar" e o aviso de conferencia vem do CODIGO (`AVISO_LLM`), nao da boa vontade
  do modelo; o LLM so escreve o resumo;
- toda saida passa por uma GUARDA (`verificar_saida`): afirmacao de compatibilidade sem ressalva,
  linguagem absoluta e codigo de item que nao estava na lista bloqueiam o texto. A guarda roda DURANTE o
  streaming: ao detectar violacao a geracao e interrompida e `Explicacao.bloqueada` fica True (a tela deve
  apagar o texto parcial e mostrar so a lista);
- prompt enxuto e estruturado: so codigo, descricao, preco e a situacao de veiculo que o proprio motor ja
  calculou (o NCM fica de fora: confundiu os modelos na medicao de 07/11).

Cliente HTTP: `requests` em /api/chat com stream=True, como `retrieval/ollama_embed.py` (sem dependencia nova).
O Ollama e o modelo sao locais; nada sai do computador.

Estado de teste: a logica (prompt, guarda, leitura do stream, falhas) tem testes com sessao falsa; o formato
real das respostas do Ollama (/api/chat, campos de duracao em nanossegundos) segue a documentacao e esta
[NAO TESTADO] contra um servidor de verdade ate rodar `scripts/12_medir_llm.py`.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd
import requests

from autopecas.retrieval.ollama_embed import _host_padrao

# Decisao 59: o resumo padrao do app e por regras (resumo.py); o LLM e complemento experimental com o
# llama3.2:3b (nao afirmou compatibilidade nem inventou item nas medicoes, mas recusa demais) e o prompt "c".
# O gemma3:1b saiu do app: afirmou sem base em 8 de 12 respostas e a guarda de texto nao pegou nenhuma.
MODELO_PADRAO = "llama3.2:3b"
VARIANTE_PADRAO = "c"
MAX_ITENS = 5
NUM_PREDICT = 120  # teto de tokens: mantem a resposta curta (o 1B variou de 72 a 140 sem teto)
TEMPERATURA = 0.1
KEEP_ALIVE = "10m"
TIMEOUT_CONEXAO = 3.0
TIMEOUT_LEITURA = 60.0  # por leitura de rede, nao total
LIMITE_TOTAL_S = 60.0  # teto de parede por explicacao (a meta e 30 s; isto e so o freio de seguranca)
LIMITE_DESCRICAO = 120
LIMITE_CONSULTA = 200

AVISO_LLM = (
    "Resumo gerado por IA local a partir dos itens listados. Sugestão a validar: "
    "confirme a aplicação no veículo antes de vender."
)

AVISO_LLM_EXPERIMENTAL = (
    "Experimental: modelos pequenos podem confundir a peça ou recusar sem motivo. A lista acima e o "
    "resumo por regras são a fonte; confira a descrição dos itens."
)

PROMPT_SISTEMA = (
    "Você ajuda o balcão de uma autopeças. Responda em português, em NO MÁXIMO 3 frases curtas, "
    "usando SOMENTE os itens listados. Cite os itens pelo código entre colchetes, exatamente como "
    "aparecem. Nunca afirme nem garanta compatibilidade com o veículo: diga o que a descrição do "
    "fornecedor traz e o que o balcão deve conferir. Não invente preços, medidas, códigos nem "
    "veículos. Se nenhum item parecer corresponder ao pedido, diga que não encontrou."
)

PROMPT_SISTEMA_B = (
    "Você ajuda o balcão de uma autopeças. Resuma em até 3 frases curtas, em português, o que a lista "
    "traz para o pedido. Cite os itens pelo código entre colchetes e use só o que está na lista "
    "(descrição, preço, veículo). Se um item cita o veículo do pedido, diga isso; se for genérico, diga "
    "que é genérico; nunca afirme que serve nem que é compatível. Só diga que não encontrou se NENHUM "
    "item da lista for a peça pedida."
)

# Exemplos sinteticos (codigos 'Z:EX..' nunca existem no catalogo: se o modelo os copiar, a guarda bloqueia).
_EXEMPLOS_B = (
    (
        "cabo de acelerador gol",
        ["[Z:EX01] CABO ACELERADOR GOL 90/95 | R$ 25,00 | veículo: cita gol",
         "[Z:EX02] CABO ACELERADOR GENERICO | R$ 18,00 | veículo: sem veículo na descrição (genérico)"],
        "O [Z:EX01] cita o Gol na descrição do fornecedor, por R$ 25,00. O [Z:EX02] é genérico, sem "
        "veículo informado, por R$ 18,00. Confira o ano e o modelo antes de vender.",
    ),
    (
        "carburador fusca",
        ["[Z:EX03] JUNTA DO CARBURADOR FUSCA | R$ 9,00 | veículo: cita fusca",
         "[Z:EX04] FILTRO DE AR FUSCA | R$ 22,00 | veículo: cita fusca"],
        "Não encontrei o carburador: os itens da lista são outras peças (junta do carburador e filtro de ar). "
        "Vale consultar outro fornecedor.",
    ),
)

PROMPT_SISTEMA_C = (
    "Você ajuda o balcão de uma autopeças. Resuma em até 3 frases curtas, em português, os itens da lista "
    "que são a peça pedida. Cite cada item pelo código entre colchetes e pelo preço, e diga se a descrição "
    "cita o veículo do pedido, cita outro veículo ou é genérica (sem veículo). Itens genéricos da peça "
    "pedida devem ser apresentados, não recusados. Nunca diga que serve nem que é compatível. Só diga que "
    "não encontrou se NENHUM item da lista for a peça pedida; nesse caso não cite preços."
)

# Variante c: a recusa vem PRIMEIRO e o exemplo "so genericos" por ultimo (o modelo pequeno imita mais o ultimo);
# a recusa nao enumera as pecas (o 1B copiava a frase inteira, inclusive o parentese, em pedidos validos).
_EXEMPLOS_C = (
    (
        "carburador fusca",
        ["[Z:EX03] JUNTA DO CARBURADOR FUSCA | R$ 9,00 | veículo: cita fusca",
         "[Z:EX04] FILTRO DE AR FUSCA | R$ 22,00 | veículo: cita fusca"],
        "Não encontrei o carburador entre os itens da lista. Vale consultar outro fornecedor.",
    ),
    _EXEMPLOS_B[0],
    (
        "retrovisor uno",
        ["[Z:EX05] RETROVISOR EXTERNO UNIVERSAL | R$ 40,00 | veículo: sem veículo na descrição (genérico)",
         "[Z:EX06] RETROVISOR EXTERNO | R$ 55,00 | veículo: sem veículo na descrição (genérico)"],
        "O [Z:EX05] por R$ 40,00 e o [Z:EX06] por R$ 55,00 são retrovisores genéricos, sem veículo "
        "informado. Nenhum cita o Uno: confira antes de vender.",
    ),
)
VARIANTES = ("a", "b", "c")
ITENS_RESUMO_C = 3  # a variante c resume so os 3 primeiros (o 3B estourava o teto de tokens com 5)


def max_itens_da_variante(variante: str) -> int:
    return ITENS_RESUMO_C if variante == "c" else MAX_ITENS


# ---------------------------------------------------------------------------- prompt


def _limpar(texto, limite: int) -> str:
    """Uma linha so, sem caracteres de controle (descricao do catalogo e dado, nao instrucao)."""
    t = re.sub(r"\s+", " ", "" if texto is None else str(texto)).strip()
    t = "".join(c for c in t if unicodedata.category(c)[0] != "C")
    return t[:limite]


def _preco(valor) -> str:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "preço n/d"
    if pd.isna(v):
        return "preço n/d"
    return "R$ " + f"{v:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


limpar_texto = _limpar  # nomes publicos para outros modulos (resumo.py)
formatar_preco = _preco


def codigo_do_item(fornecedor, codigo_interno) -> str:
    return f"{_limpar(fornecedor, 4)}:{_limpar(codigo_interno, 40)}"


def itens_para_prompt(itens: pd.DataFrame, max_itens: int = MAX_ITENS) -> list[dict]:
    """Os primeiros `max_itens` da lista, so com o que o LLM precisa (sem NCM, sem GTIN)."""
    saida = []
    tem_situacao = "situacao_veiculo" in itens.columns
    for _, linha in itens.head(max_itens).iterrows():
        saida.append({
            "codigo": codigo_do_item(linha["fornecedor"], linha["codigo_interno"]),
            "descricao": _limpar(linha["descricao"], LIMITE_DESCRICAO),
            "preco": _preco(linha["preco"]),
            "veiculo": _limpar(linha["situacao_veiculo"], 80) if tem_situacao else "",
        })
    return saida


def _usuario(pedido: str, linhas_item: list[str]) -> str:
    return f"Pedido do balcão: {pedido}\n\nItens do catálogo:\n" + "\n".join(linhas_item)


def _prefixo_mensagens(variante: str) -> list[dict]:
    """Parte fixa do prompt (instrucao de sistema + exemplos): igual em toda chamada da variante."""
    if variante == "a":
        return [{"role": "system", "content": PROMPT_SISTEMA}]
    sistema, exemplos = (PROMPT_SISTEMA_B, _EXEMPLOS_B) if variante == "b" else (PROMPT_SISTEMA_C, _EXEMPLOS_C)
    mensagens = [{"role": "system", "content": sistema}]
    for pedido_ex, itens_ex, resposta_ex in exemplos:
        mensagens.append({"role": "user", "content": _usuario(pedido_ex, itens_ex)})
        mensagens.append({"role": "assistant", "content": resposta_ex})
    return mensagens


def montar_mensagens(
    consulta: str, itens: pd.DataFrame, max_itens: int | None = None, variante: str = "a"
) -> tuple[list[dict], frozenset]:
    """(mensagens para /api/chat, conjunto de codigos permitidos na resposta).

    variante "a": so a instrucao de sistema (a de 07/11). "b": instrucao mais curta e 2 exemplos em turnos
    de chat (um com resposta, um de "nao encontrei"), para calibrar a recusa dos modelos pequenos.
    """
    if variante not in VARIANTES:
        raise ValueError(f"variante de prompt desconhecida: {variante!r}")
    if max_itens is None:
        max_itens = max_itens_da_variante(variante)
    pedido = _limpar(consulta, LIMITE_CONSULTA)
    if not pedido:
        raise ValueError("consulta vazia")
    linhas = itens_para_prompt(itens, max_itens)
    if not linhas:
        raise ValueError("sem itens para explicar")
    corpo = []
    for it in linhas:
        txt = f"[{it['codigo']}] {it['descricao']} | {it['preco']}"
        if it["veiculo"]:
            txt += f" | veículo: {it['veiculo']}"
        corpo.append(txt)
    mensagens = _prefixo_mensagens(variante)
    mensagens.append({"role": "user", "content": _usuario(pedido, corpo)})
    return mensagens, frozenset(it["codigo"] for it in linhas)


# ---------------------------------------------------------------------------- guarda


@dataclass(frozen=True)
class Violacao:
    tipo: str  # "absoluto", "afirmacao" ou "codigo_inventado"
    trecho: str


# Linguagem absoluta: bloqueia sempre. Cada padrao em linha propria.
_ABSOLUTOS = (
    r"\b100\s*%",
    r"\bcom certeza\b",
    r"\bsem duvida\b",
    r"\bperfeitamente\b",
)
# Afirmacoes: bloqueiam so se NAO houver ressalva nas 6 palavras anteriores da mesma frase.
_AFIRMATIVOS_RAIZ = ("compativ",)
# "garantir" (infinitivo, conselho: "e importante garantir que...") NAO entra: so as formas que prometem.
_AFIRMATIVOS_PALAVRA = frozenset({
    "serve", "servem", "encaixa", "encaixam", "ideal", "ideais",
    "garante", "garantem", "garantido", "garantida", "garantidos", "garantidas",
})
_RESSALVAS = frozenset({
    "nao", "talvez", "possivel", "possivelmente", "pode", "podem", "poderia", "confirmar", "confirme",
    "confirma", "verificar", "verifique", "validar", "valide", "validacao", "conferir", "confira",
    "checar", "cheque", "duvida", "questionavel", "se", "caso", "antes", "sugestao",
})
_JANELA_RESSALVA = 6
_CODIGO = re.compile(r"\b([A-Z]):([A-Za-z0-9][A-Za-z0-9._\-/]*)")


def _sem_acento(texto: str) -> str:
    base = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in base if not unicodedata.combining(c))


def verificar_saida(texto: str, codigos_permitidos, final: bool = False) -> list[Violacao]:
    """Violacoes no texto do LLM. `final=False` (durante o streaming) ignora o codigo cortado no fim do texto."""
    violacoes: list[Violacao] = []
    norm = _sem_acento(texto)

    for padrao in _ABSOLUTOS:
        achado = re.search(padrao, norm)
        if achado:
            violacoes.append(Violacao("absoluto", achado.group(0)))

    for frase in re.split(r"[.!?;\n]", norm):
        palavras = re.findall(r"[a-z0-9]+", frase)
        for i, p in enumerate(palavras):
            afirma = p in _AFIRMATIVOS_PALAVRA or any(p.startswith(r) for r in _AFIRMATIVOS_RAIZ)
            if afirma and not _RESSALVAS.intersection(palavras[max(0, i - _JANELA_RESSALVA):i]):
                violacoes.append(Violacao("afirmacao", " ".join(palavras[max(0, i - 3):i + 1])))

    permitidos = set(codigos_permitidos)
    for m in _CODIGO.finditer(texto):
        if not final and m.end() == len(texto):
            continue  # ainda pode estar chegando (ex.: "A:000" antes de "A:00001")
        codigo = f"{m.group(1)}:{m.group(2).rstrip('._-/')}"
        if codigo not in permitidos:
            violacoes.append(Violacao("codigo_inventado", codigo))
    return violacoes


# ---------------------------------------------------------------------------- cliente


@dataclass(frozen=True)
class Estatisticas:
    """Campos de duracao do ultimo pedaco do /api/chat (o Ollama os devolve em nanossegundos)."""

    carga_s: float | None
    prompt_tokens: int | None
    prompt_s: float | None
    gerados: int | None
    geracao_s: float | None
    total_s: float | None

    @property
    def eval_rate(self) -> float | None:
        if self.gerados and self.geracao_s:
            return self.gerados / self.geracao_s
        return None

    @property
    def prompt_rate(self) -> float | None:
        if self.prompt_tokens and self.prompt_s:
            return self.prompt_tokens / self.prompt_s
        return None


def _segundos(ns) -> float | None:
    return float(ns) / 1e9 if isinstance(ns, (int, float)) else None


def _estatisticas(dado: dict) -> Estatisticas:
    return Estatisticas(
        carga_s=_segundos(dado.get("load_duration")),
        prompt_tokens=dado.get("prompt_eval_count"),
        prompt_s=_segundos(dado.get("prompt_eval_duration")),
        gerados=dado.get("eval_count"),
        geracao_s=_segundos(dado.get("eval_duration")),
        total_s=_segundos(dado.get("total_duration")),
    )


def indica_recusa(texto: str) -> bool:
    """O texto diz que nao encontrou a peca (sem acento nem caixa). Indicador de recusa, nao de qualidade."""
    base = "".join(c for c in unicodedata.normalize("NFKD", (texto or "").lower()) if not unicodedata.combining(c))
    return "nao encontr" in base


def _ate_ultima_frase(texto: str) -> str:
    """Texto cortado no teto de tokens: fica ate a ultima frase completa (ponto seguido de espaco ou fim).
    Sem nenhuma frase completa, mantem o texto e marca o corte com reticencias."""
    fins = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", texto)]
    if fins:
        return texto[: fins[-1]].strip()
    return texto.rstrip(" ,;:") + "…"


@dataclass
class Explicacao:
    """Uma geracao em streaming. Use como iterador de pedacos de texto (serve a `st.write_stream`).

    Depois de consumida: `texto_final` (vazio se bloqueada ou com erro), `bloqueada`, `violacoes`, `erro`,
    `t_primeira_palavra` e `t_total` (s, relogio do cliente) e `estatisticas` (as do servidor).
    Nunca levanta excecao por falha do Ollama: guarda a mensagem em `erro`.
    """

    mensagens: list[dict]
    codigos: frozenset
    modelo: str = MODELO_PADRAO
    host: str | None = None
    sessao: object = None
    num_predict: int = NUM_PREDICT
    temperatura: float = TEMPERATURA
    keep_alive: str = KEEP_ALIVE
    limite_total_s: float = LIMITE_TOTAL_S
    relogio: object = time.perf_counter
    # None = nao envia `think` (modelos sem raciocinio, como gemma3 e llama3.2, ficam como estao);
    # False = pede ao Ollama para NAO raciocinar (modelos com a marca "thinking"). [NAO TESTADO] no servidor real.
    pensar: bool | None = None
    texto: str = field(default="", init=False)
    chars_pensamento: int = field(default=0, init=False)
    erro: str | None = field(default=None, init=False)
    bloqueada: bool = field(default=False, init=False)
    violacoes: list = field(default_factory=list, init=False)
    estatisticas: Estatisticas | None = field(default=None, init=False)
    t_primeira_palavra: float | None = field(default=None, init=False)
    t_total: float | None = field(default=None, init=False)
    concluida: bool = field(default=False, init=False)
    truncada: bool = field(default=False, init=False)  # parou no teto de tokens (done_reason == "length")
    _consumida: bool = field(default=False, init=False, repr=False)

    @property
    def texto_final(self) -> str:
        if self.bloqueada or self.erro:
            return ""
        texto = self.texto.strip()
        return _ate_ultima_frase(texto) if self.truncada else texto

    def __iter__(self) -> Iterator[str]:
        if self._consumida:
            return iter(())
        self._consumida = True
        return self._executar()

    def _falhar(self, mensagem: str) -> None:
        self.erro = mensagem

    def _executar(self) -> Iterator[str]:
        sessao = self.sessao or requests.Session()
        url = f"{(self.host or _host_padrao()).rstrip('/')}/api/chat"
        corpo = {
            "model": self.modelo,
            "messages": self.mensagens,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperatura, "num_predict": self.num_predict},
        }
        if self.pensar is not None:
            corpo["think"] = self.pensar
        t0 = self.relogio()
        resp = None
        try:
            try:
                resp = sessao.post(url, json=corpo, timeout=(TIMEOUT_CONEXAO, TIMEOUT_LEITURA), stream=True)
            except (requests.ConnectionError, requests.Timeout):
                self._falhar("Não consegui falar com o Ollama (servidor fora do ar).")
                return
            if resp.status_code == 404:
                self._falhar(f"Modelo '{self.modelo}' não encontrado no Ollama (ollama pull {self.modelo}).")
                return
            if resp.status_code != 200:
                self._falhar(f"O Ollama respondeu HTTP {resp.status_code}.")
                return
            try:
                for bruto in resp.iter_lines():
                    if self.relogio() - t0 > self.limite_total_s:
                        self._falhar("Tempo esgotado ao gerar o resumo.")
                        return
                    if not bruto:
                        continue
                    linha = bruto.decode("utf-8", errors="replace") if isinstance(bruto, bytes) else bruto
                    try:
                        dado = json.loads(linha)
                    except ValueError:
                        continue
                    if dado.get("error"):
                        self._falhar("O Ollama devolveu um erro ao gerar o resumo.")
                        return
                    mensagem = dado.get("message") or {}
                    self.chars_pensamento += len(mensagem.get("thinking") or "")
                    pedaco = mensagem.get("content", "")
                    if pedaco:
                        if self.t_primeira_palavra is None:
                            self.t_primeira_palavra = self.relogio() - t0
                        self.texto += pedaco
                        achadas = verificar_saida(self.texto, self.codigos)
                        if achadas:
                            self.bloqueada, self.violacoes = True, achadas
                            return
                        yield pedaco
                    if dado.get("done"):
                        self.estatisticas = _estatisticas(dado)
                        self.truncada = dado.get("done_reason") == "length"
                        self.concluida = True
                        break
            except requests.RequestException:
                self._falhar("A conexão com o Ollama foi interrompida.")
                return
            if not self.texto:
                if self.chars_pensamento:
                    self._falhar("O modelo só gerou raciocínio, sem resposta (tente com o raciocínio desligado).")
                else:
                    self._falhar("O Ollama não devolveu texto.")
            elif self.concluida:
                achadas = verificar_saida(self.texto, self.codigos, final=True)
                if achadas:
                    self.bloqueada, self.violacoes = True, achadas
            else:
                self._falhar("Resposta incompleta do Ollama.")
        finally:
            self.t_total = self.relogio() - t0
            if resp is not None and hasattr(resp, "close"):
                resp.close()


def explicar(consulta: str, itens: pd.DataFrame, **kw) -> Explicacao | None:
    """Prepara a explicacao dos primeiros itens da lista. None se nao ha nada a explicar."""
    try:
        mensagens, codigos = montar_mensagens(
            consulta, itens, kw.pop("max_itens", None), kw.pop("variante", VARIANTE_PADRAO)
        )
    except ValueError:
        return None
    return Explicacao(mensagens=mensagens, codigos=codigos, **kw)


# ---------------------------------------------------------------------------- servidor


def modelo_disponivel(modelo: str, host: str | None = None, sessao=None, timeout: float = 2.0) -> bool:
    """O modelo esta instalado? Sonda direta (GET /api/tags); comparacao EXATA do nome (1b != 4b)."""
    sessao = sessao or requests.Session()
    try:
        resp = sessao.get(f"{(host or _host_padrao()).rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return False
        nomes = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception:  # sonda de saude: qualquer falha significa "indisponivel"
        return False
    return modelo in nomes or (":" not in modelo and f"{modelo}:latest" in nomes)


def descarregar(modelo: str, host: str | None = None, sessao=None, timeout: float = 10.0) -> bool:
    """Tira o modelo da RAM (keep_alive 0), para medir um modelo por vez com memoria limpa."""
    sessao = sessao or requests.Session()
    try:
        resp = sessao.post(
            f"{(host or _host_padrao()).rstrip('/')}/api/generate",
            json={"model": modelo, "keep_alive": 0}, timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def aquecer_modelo(
    modelo: str, host: str | None = None, sessao=None, timeout: float = 120.0, variante: str = VARIANTE_PADRAO
) -> bool:
    """Carrega o modelo e processa o prefixo fixo do prompt (instrucao + exemplos), gerando 1 token.

    Medido: a 1a geracao depois de so carregar o llama3.2:3b levou 14,5 s ate a primeira palavra no app (e 13,5 a
    22,6 s no script), contra 2 a 3 s nas seguintes, porque o prefixo de ~600 tokens ainda nao tinha sido
    processado. Aqui ele e processado uma vez, antes do clique. True se o Ollama respondeu 200.
    """
    sessao = sessao or requests.Session()
    mensagens = _prefixo_mensagens(variante) + [
        {"role": "user", "content": _usuario("aquecimento", ["[Z:EX00] ITEM DE AQUECIMENTO | R$ 1,00"])}
    ]
    corpo = {
        "model": modelo, "messages": mensagens, "stream": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": TEMPERATURA, "num_predict": 1},
    }
    try:
        resp = sessao.post(f"{(host or _host_padrao()).rstrip('/')}/api/chat", json=corpo, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
