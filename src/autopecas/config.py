"""Configuracoes centrais: caminhos, colunas e niveis de escala."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "resultados"

ARQUIVO_XLSX = DADOS_BRUTOS / "BD_Proj_Integrador_5CD.xlsx"
CATALOGO_CSV = DADOS_PROCESSADOS / "catalogo.csv"

# A planilha original NAO tem cabecalho: esta e a ordem das colunas A-F.
COLUNAS = ["codigo_interno", "gtin", "descricao", "preco", "ncm", "fornecedor"]

# Niveis do escopo (acumulativos). None = todos os fornecedores presentes.
# O fornecedor M nao existe na planilha (13 fornecedores: A-L e N).
NIVEIS = {
    "Básico": tuple("ABCD"),
    "Intermediário": tuple("ABCDEFGHI"),
    "Avançado": None,
}
