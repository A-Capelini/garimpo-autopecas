"""Passo 3 - ablacao do baseline lexical (normalizacao x min_df x sublinear_tf).

  python scripts/03_ablacao_lexical.py
Grava resultados/ablacao_lexical.csv (sobrescreve) e imprime Hit@1/Hit@10/MRR por celula.
"""
from autopecas.config import RESULTADOS
from autopecas.data.loader import carregar_catalogo
from autopecas.evaluation.ablacao import ablacao_lexical


def main() -> None:
    df = carregar_catalogo()
    tab = ablacao_lexical(df)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / "ablacao_lexical.csv"
    tab.to_csv(destino, index=False)
    for nivel in ("Intermediário", "Avançado"):
        print(f"\n== {nivel} ==")
        sub = tab[tab["nivel"] == nivel][["normalizacao", "min_df", "sublinear_tf", "hit@1", "hit@5", "hit@10", "mrr"]]
        print(sub.round(4).to_string(index=False))
    print(f"\nSalvo em {destino}")


if __name__ == "__main__":
    main()
