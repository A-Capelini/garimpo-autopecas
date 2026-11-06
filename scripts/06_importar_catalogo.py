"""Passo 6 - importa a planilha no catalogo SQLite (upsert por fornecedor + codigo).

Uso:
  python scripts/06_importar_catalogo.py                       # data/raw/BD_Proj_Integrador_5CD.xlsx -> data/catalogo.db
  python scripts/06_importar_catalogo.py --sincronizar         # itens que sumiram da planilha ficam inativos
  python scripts/06_importar_catalogo.py --absorver-cache data/processed/embeddings.db   # reaproveita vetores da Etapa 1
"""
import argparse
from pathlib import Path

from autopecas.config import ARQUIVO_XLSX, CATALOGO_DB
from autopecas.data.loader import carregar_xlsx
from autopecas.data.quality import aplicar_qualidade
from autopecas.store import Catalogo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default=str(ARQUIVO_XLSX), help="planilha consolidada (sem cabecalho)")
    ap.add_argument("--db", default=str(CATALOGO_DB), help="arquivo SQLite do catalogo")
    ap.add_argument("--sincronizar", action="store_true",
                    help="desativa itens dos fornecedores da planilha que nao estao nela")
    ap.add_argument("--absorver-cache", metavar="EMBEDDINGS_DB",
                    help="copia vetores de um embeddings.db antigo para o catalogo")
    args = ap.parse_args()

    df = aplicar_qualidade(carregar_xlsx(args.xlsx))
    with Catalogo(args.db) as cat:
        if args.absorver_cache:
            n = cat.absorver_cache(Path(args.absorver_cache))
            print(f"Vetores absorvidos de {args.absorver_cache}: {n}")
        relatorio = cat.importar(df, sincronizar=args.sincronizar)
        print(f"Importacao ({len(df)} linhas da planilha): {relatorio.resumo()}")
        print(f"Catalogo {args.db}: {cat.contar()} | vetores no arquivo: {cat.contar_vetores()}")


if __name__ == "__main__":
    main()
