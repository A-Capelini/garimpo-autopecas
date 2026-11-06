#!/usr/bin/env bash
# Medicao da Etapa 3: gabarito manual do nivel Basico (perguntas em texto livre).
# Uso: bash scripts/09_medicao_etapa3.sh [arquivo.csv]   (padrao: data/raw/gabarito_basico.csv)
# 1) confere o arquivo contra o catalogo; 2) mede lexical, bge-m3 e hibrido escolhido na mesma
# execucao -> resultados/avaliacao_texto.csv + resultados/revisao_basico_*.md (fora do Git).
# Precisa do Ollama no ar para o bge-m3; os vetores do Basico costumam estar no cache da Etapa 1.
set -euo pipefail
cd "$(dirname "$0")/.."
gab="${1:-data/raw/gabarito_basico.csv}"
python scripts/02_avaliar.py --retriever lexical --gabarito "$gab" --so-validar
python scripts/02_avaliar.py --retriever todos --gabarito "$gab"
echo
echo "Resumo em resultados/avaliacao_texto.csv; revisao manual em resultados/revisao_basico_*.md"
