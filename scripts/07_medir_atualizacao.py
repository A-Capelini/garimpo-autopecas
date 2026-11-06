"""Passo 7 - mede o custo de adicionar itens novos ao catalogo (precisa do Ollama no ar).

Usa um catalogo TEMPORARIO; o data/catalogo.db real nao e tocado.
Uso: python scripts/07_medir_atualizacao.py [--novos 100] [--cache-origem data/processed/embeddings.db]
"""
import argparse

from autopecas.config import ARQUIVO_XLSX, DADOS_PROCESSADOS, RESULTADOS
from autopecas.data.loader import carregar_xlsx
from autopecas.data.quality import aplicar_qualidade
from autopecas.store.medicao import formatar_markdown, medir_atualizacao


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(ARQUIVO_XLSX))
    ap.add_argument("--novos", type=int, default=100)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--modelo", default="bge-m3")
    ap.add_argument("--cache-origem", default=str(DADOS_PROCESSADOS / "embeddings.db"),
                    help="embeddings.db da Etapa 1: evita embedar a base inteira (~10 min)")
    args = ap.parse_args()

    df = aplicar_qualidade(carregar_xlsx(args.xlsx))
    m = medir_atualizacao(df, n_novos=args.novos, semente=args.semente,
                          cache_origem=args.cache_origem, modelo=args.modelo)
    texto = formatar_markdown(m, n_total=len(df))
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / "medicao_atualizacao.md"
    destino.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Salvo em {destino}")


if __name__ == "__main__":
    main()
