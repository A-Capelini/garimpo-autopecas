"""Passo 11 - holdout: sugere pecas esperadas por filtro de palavras-chave e fecha o gabarito apos a conferencia.

NAO usa o ranking do sistema (decisao 33). Nao mede nada e nao precisa do Ollama.

Uso (a partir da raiz do repositorio, com o ambiente garimpo-autopecas ativo):
  # 1) gera a planilha de conferencia + relatorio de existencia de pecas (Basico e Intermediario)
  python scripts/11_sugerir_pecas_holdout.py
  python scripts/11_sugerir_pecas_holdout.py --nivel "Intermediário"

  # 2) depois que o grupo preencher S/N na planilha: valida e grava um gabarito por nivel
  #    (a conferencia feita no Intermediario gera tambem o gabarito do Basico: so conta os fornecedores do nivel)
  python scripts/11_sugerir_pecas_holdout.py --fechar resultados/conferencia_holdout_intermediario.xlsx \\
      --nivel "Intermediário" --gabarito-saida data/raw/holdout_gabarito_intermediario.csv
  python scripts/11_sugerir_pecas_holdout.py --fechar resultados/conferencia_holdout_intermediario.xlsx \\
      --nivel "Básico" --gabarito-saida data/raw/holdout_gabarito_basico.csv

Entradas: data/raw/holdout_perguntas.csv (id;pergunta;tipo;autor;observacao) e o catalogo (CATALOGO_CSV).
Saidas em resultados/ (fora do Git): conferencia_holdout_<nivel>.xlsx, sugestoes_holdout_<nivel>.csv,
relatorio_holdout_<nivel>.md. Com --fechar: o CSV de --gabarito-saida (fora do Git), lido por
scripts/10_medir_veiculo.py --gabarito.
"""
import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from autopecas.config import CATALOGO_CSV, NIVEIS, RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.evaluation.conferencia_holdout import escrever_conferencia, ler_conferencia
from autopecas.evaluation.gabarito_texto import ErroGabarito, ler_gabarito, resolver, itens_do_nivel
from autopecas.evaluation.sugestao_holdout import gabarito_de_conferencia, montar_indice, sugerir


def _slug(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _ler_holdout(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    faltam = [c for c in ("id", "pergunta", "tipo") if c not in df.columns]
    if faltam:
        raise SystemExit(f"Faltam colunas no holdout: {', '.join(faltam)} (lidas: {', '.join(df.columns)}).")
    if df["id"].duplicated().any():
        raise SystemExit(f"ids repetidos no holdout: {sorted(df.loc[df['id'].duplicated(), 'id'])}")
    return df


def _catalogo(caminho: Path) -> pd.DataFrame:
    return carregar_catalogo(caminho)


def _relatorio(nivel, sha, sugestoes_por_nivel, holdout, tipos) -> str:
    n = len(holdout)
    linhas = [
        f"# Holdout: existência de peças por filtro de palavras-chave (triagem automática)",
        "",
        f"- Perguntas: {n} (holdout sha256 `{sha}…`). Nível da planilha de conferência: **{nivel}**.",
        "- **Isto é triagem, não gabarito.** Vale o que o grupo marcar S/N na planilha (decisões 36 e 45).",
        "- O filtro não usa o ranking do sistema (decisão 33). `provável` = casa todos os termos e, com veículo na "
        "pergunta, cita o veículo; `possível` = casa parte dos termos, casou por grafia ou há termo sem correspondente; `só improvável` = há item no filtro, mas o critério o descarta (genérico ou de outro veículo, lado errado). Prováveis podem incluir peça parecida mas diferente (ex.: capa da correia dentada ≠ correia dentada): só a leitura do grupo decide.",
        "",
        "## Perguntas com peça, por nível",
        "",
        "| Nível | Itens | Com provável | Só possível | Só improvável | Nada no filtro |",
        "|---|---|---|---|---|---|",
    ]
    for nome, (indice, sugs) in sugestoes_por_nivel.items():
        c = pd.Series([s.existe_peca for s in sugs]).value_counts()
        linhas.append(f"| {nome} | {len(indice.itens)} | {c.get('sim', 0)} | {c.get('talvez', 0)} | "
                      f"{c.get('so_improvavel', 0)} | {c.get('nada', 0)} |")
    linhas += ["", "## Por tipo de pergunta (nível da planilha)", "", "| Tipo | Perguntas | Com provável | Só possível | Só improvável | Nada no filtro |", "|---|---|---|---|---|---|"]
    sugs = sugestoes_por_nivel[nivel][1]
    tab = pd.DataFrame({"tipo": [tipos[s.id] for s in sugs], "x": [s.existe_peca for s in sugs]})
    for tipo, g in tab.groupby("tipo"):
        c = g["x"].value_counts()
        linhas.append(f"| {tipo} | {len(g)} | {c.get('sim', 0)} | {c.get('talvez', 0)} | "
                      f"{c.get('so_improvavel', 0)} | {c.get('nada', 0)} |")
    linhas += ["", "## As `sem_resposta` do Gemini (devem não ter peça)", ""]
    for s in sugs:
        if tipos[s.id] == "sem_resposta":
            achou = f"{s.contagem.get('provavel', 0)} provável, {s.contagem.get('possivel', 0)} possível"
            top = f" — ex.: {s.candidatas[0].fornecedor}:{s.candidatas[0].codigo} {s.candidatas[0].descricao}" if s.candidatas else ""
            linhas.append(f"- {s.id} «{s.pergunta}»: {achou}{top}")
    linhas += ["", "## Termos sem correspondente no catálogo (conferir sinônimo)", ""]
    for s in sugs:
        if s.sem_correspondencia:
            linhas.append(f"- {s.id} «{s.pergunta}»: {', '.join(s.sem_correspondencia)}")
    return "\n".join(linhas) + "\n"


def gerar(args) -> None:
    holdout = _ler_holdout(args.holdout)
    sha = hashlib.sha256(args.holdout.read_bytes()).hexdigest()[:12]
    df = _catalogo(args.catalogo)
    tipos = dict(zip(holdout["id"], holdout["tipo"]))

    por_nivel = {}
    for nome in dict.fromkeys(["Básico", "Intermediário", args.nivel]):
        indice = montar_indice(itens_do_nivel(df, nome), nome)
        sugs = [sugerir(i, p, indice) for i, p in zip(holdout["id"], holdout["pergunta"])]
        por_nivel[nome] = (indice, sugs)
        print(f">> {nome}: {len(indice.itens)} itens, {len(sugs)} perguntas filtradas", file=sys.stderr)

    nivel = args.nivel
    sugs = por_nivel[nivel][1]
    args.saida.mkdir(parents=True, exist_ok=True)
    rotulo = _slug(nivel)
    xlsx = args.saida / f"conferencia_holdout_{rotulo}.xlsx"
    escrever_conferencia(xlsx, sugs, tipos, nivel, sha)
    linhas = [{
        "id": s.id, "pergunta": s.pergunta, "tipo": tipos[s.id], "veiculo": ",".join(s.veiculos), "ano": s.ano or "",
        "termos": " ".join(s.termos), "sem_correspondencia": " ".join(s.sem_correspondencia),
        "total_filtro": s.total_filtro, "provavel": s.contagem.get("provavel", 0),
        "possivel": s.contagem.get("possivel", 0), "improvavel": s.contagem.get("improvavel", 0),
        "existe_peca": s.existe_peca,
    } for s in sugs]
    pd.DataFrame(linhas).to_csv(args.saida / f"sugestoes_holdout_{rotulo}.csv", index=False)
    rel = args.saida / f"relatorio_holdout_{rotulo}.md"
    rel.write_text(_relatorio(nivel, sha, por_nivel, holdout, tipos), encoding="utf-8")
    print(f"Gerados em {args.saida}: {xlsx.name}, sugestoes_holdout_{rotulo}.csv, {rel.name}")
    print(rel.read_text(encoding="utf-8"))


def fechar(args) -> None:
    holdout = _ler_holdout(args.holdout)
    revisadas, extras = ler_conferencia(args.fechar)
    letras = NIVEIS[args.nivel]
    gab, pendencias = gabarito_de_conferencia(
        holdout, revisadas, extras, frozenset(letras) if letras else None)
    if pendencias:
        print(f"\n{len(pendencias)} pendência(s) na conferência; o gabarito NÃO foi gravado:")
        for p in pendencias[:60]:
            print(" -", p)
        if len(pendencias) > 60:
            print(f" ... e mais {len(pendencias) - 60}")
        raise SystemExit(1)
    destino = args.gabarito_saida
    destino.parent.mkdir(parents=True, exist_ok=True)
    gab.to_csv(destino, sep=";", index=False, encoding="utf-8")
    try:  # confere o arquivo como o script 10 vai ler
        perguntas = ler_gabarito(destino)
        resolver(perguntas, _catalogo(args.catalogo), args.nivel)
    except ErroGabarito as e:
        raise SystemExit(f"\nO gabarito gravado em {destino} não passa na validação:\n{e}")
    sem = int((gab["tipo"] == "sem_resposta").sum())
    sem_tipo = int((gab["tipo"] == "").sum())
    print(f"Gabarito gravado em {destino}: {len(gab)} perguntas; {len(gab) - sem} com peça; {sem} sem_resposta.")
    if sem_tipo:
        print(f"ATENÇÃO: {sem_tipo} pergunta(s) tinham tipo sem_resposta e agora têm peça; sem tipo (nao_informado). "
              "Escolha o tipo no CSV antes de medir.")
    print(f"sha256: {hashlib.sha256(destino.read_bytes()).hexdigest()[:12]}…")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", type=Path, default=Path("data/raw/holdout_perguntas.csv"))
    ap.add_argument("--catalogo", type=Path, default=CATALOGO_CSV)
    ap.add_argument("--nivel", choices=list(NIVEIS), default="Básico")
    ap.add_argument("--saida", type=Path, default=RESULTADOS)
    ap.add_argument("--fechar", type=Path, metavar="PLANILHA", help="le a conferencia preenchida e grava o gabarito")
    ap.add_argument("--gabarito-saida", type=Path, default=Path("data/raw/holdout_gabarito.csv"),
                    help="com --fechar: use um arquivo por nivel (ex.: holdout_gabarito_basico.csv)")
    args = ap.parse_args()
    fechar(args) if args.fechar else gerar(args)


if __name__ == "__main__":
    main()
