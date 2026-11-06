"""Passo 4 - calibra o hibrido (divisao por GTIN: calibracao x teste) e decide a adocao.

  python scripts/04_calibrar_hibrido.py --modelo bge-m3
Precisa do Ollama no ar (so para embedar textos ainda fora do cache SQLite).
Grava resultados/calibracao.csv (todas as configuracoes, nos dois splits).
"""
import argparse

import pandas as pd

from autopecas.config import RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.evaluation.calibracao import COLS_METRICA, calibrar
from autopecas.retrieval.ollama_embed import ErroOllama, OllamaEmbeddingRetriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="bge-m3")
    ap.add_argument("--nivel-selecao", default="Avançado", choices=["Intermediário", "Avançado"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df = carregar_catalogo()
    try:
        tab, res = calibrar(
            df, lambda: OllamaEmbeddingRetriever(args.modelo), nivel_selecao=args.nivel_selecao, n_boot=args.n_boot
        )
    except ErroOllama as e:
        raise SystemExit(f"\nERRO: {e}")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTADOS / "calibracao.csv", index=False)
    pd.set_option("display.width", 200)

    sel = args.nivel_selecao
    cal = tab[(tab["nivel"] == sel) & (tab["split"] == "calibracao")]
    print(f"== Calibracao ({sel}): lexicais e semantico ==")
    print(cal[cal["config"].isin(["lexical", "semantico"])][["normalizacao", "min_df", "config"] + COLS_METRICA]
          .round(4).to_string(index=False))
    print(f"\n== Calibracao ({sel}): 10 melhores hibridos por MRR ==")
    hib = cal[~cal["config"].isin(["lexical", "semantico"])]
    print(hib.sort_values(["mrr", "hit@1"], ascending=False).head(10)[["normalizacao", "min_df", "config"] + COLS_METRICA]
          .round(4).to_string(index=False))

    l, h = res["lexical_escolhido"], res["hibrido_escolhido"]
    print(f"\nLexical escolhido : normalizacao={l['normalizacao']}, min_df={int(l['min_df'])}")
    print(f"Hibrido escolhido : normalizacao={h['normalizacao']}, min_df={int(h['min_df'])}, fusao={h['config']}")
    print("\n== TESTE (consultas nao usadas na escolha) ==")
    for nivel, r in res["por_nivel"].items():
        print(f"\n{nivel} (n={r['lexical']['n_consultas']} consultas de teste)")
        linhas = pd.DataFrame([{"modelo": "lexical", **r["lexical"]}, {"modelo": "hibrido", **r["hibrido"]}])
        print(linhas[["modelo"] + COLS_METRICA[1:]].round(4).to_string(index=False))
        print(f"  delta Hit@1 = {r['delta_hit@1']:+.4f}  IC95% [{r['ic95_hit@1'][0]:+.4f}, {r['ic95_hit@1'][1]:+.4f}]")
        print(f"  delta MRR   = {r['delta_mrr']:+.4f}  IC95% [{r['ic95_mrr'][0]:+.4f}, {r['ic95_mrr'][1]:+.4f}]")
    print(f"\nRegra (teste de {sel}: hibrido nao pode perder em Hit@1 nem em MRR): "
          f"{'ADOTAR HIBRIDO' if res['adotar_hibrido'] else 'MANTER LEXICAL'}")
    print(f"Detalhes em {RESULTADOS / 'calibracao.csv'}")


if __name__ == "__main__":
    main()
