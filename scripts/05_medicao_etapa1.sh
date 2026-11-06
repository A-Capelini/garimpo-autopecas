#!/usr/bin/env bash
# Medicao final da Etapa 1: lexical, semantica e hibrida nos 3 niveis -> resultados/avaliacao.csv
# Precisa do Ollama no ar (os embeddings ja estao no cache SQLite: leva segundos por execucao).
set -euo pipefail
cd "$(dirname "$0")/.."
destino=resultados/avaliacao.csv
if [ -f "$destino" ]; then
  mv "$destino" "resultados/avaliacao_anterior_$(date +%Y%m%d_%H%M%S).csv"
fi
A="python scripts/02_avaliar.py"
H="--retriever hibrido --modelo bge-m3"

# lexical
$A --retriever lexical --normalizacao oficial --min-df 2                    # baseline anterior (Etapa 0)
$A --retriever lexical --normalizacao simples --min-df 1 --sem-sublinear    # lexical do Vitor
$A --retriever lexical                                                      # lexical escolhido (padrao)
# semantica
$A --retriever ollama --modelo bge-m3 --normalizacao oficial
$A --retriever ollama --modelo bge-m3                                       # padrao: normalizacao simples
# hibrida
$A $H --normalizacao oficial --min-df 2 --k-rrf 60                          # RRF 3:1 original, texto oficial
$A $H --sem-sublinear --k-rrf 60                                            # reproduz o hibrido do Vitor
$A $H --k-rrf 1                                                             # hibrido escolhido (calibracao)
echo
echo "Resumo em $destino"
