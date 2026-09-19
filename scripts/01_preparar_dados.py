"""Passo 1 - le a planilha bruta, aplica qualidade e gera:
   data/processed/catalogo.csv  e  resultados/relatorio_qualidade.md
Uso:  python scripts/01_preparar_dados.py
"""
from autopecas.config import CATALOGO_CSV, NIVEIS, RESULTADOS
from autopecas.data.levels import fornecedores_do_nivel, mascara_nivel
from autopecas.data.loader import carregar_xlsx
from autopecas.data.quality import aplicar_qualidade, resumo_qualidade
from autopecas.evaluation.gold import construir_gold


def main() -> None:
    df = aplicar_qualidade(carregar_xlsx())
    CATALOGO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOGO_CSV, index=False)

    r = resumo_qualidade(df)
    linhas = [
        "# Relatorio de qualidade dos dados",
        "",
        f"- Linhas: **{r['linhas']}**  |  Fornecedores: **{len(r['fornecedores'])}** ({', '.join(r['fornecedores'])})",
        f"- NCMs distintos: {r['ncm_distintos']}  |  Linhas com descricao repetida: {r['descricoes_repetidas']}",
        f"- Registros com preco 0: {r['preco_zero']}",
        "",
        "## Situacao do GTIN",
        "",
        "| status | linhas | % |",
        "|---|---:|---:|",
    ]
    for status, n in r["gtin_status"].items():
        linhas.append(f"| {status} | {n} | {n / r['linhas']:.1%} |")

    linhas += ["", "## Niveis de escala e gabarito", "",
               "| nivel | fornecedores | linhas | consultas do gabarito |", "|---|---|---:|---:|"]
    for nivel in NIVEIS:
        m = mascara_nivel(df, nivel)
        g = construir_gold(df, m)
        linhas.append(
            f"| {nivel} | {'-'.join(fornecedores_do_nivel(df, nivel))} | {int(m.sum())} | {len(g)} |"
        )

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / "relatorio_qualidade.md"
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    print(f"\nCatalogo salvo em {CATALOGO_CSV}\nRelatorio salvo em {destino}")


if __name__ == "__main__":
    main()
