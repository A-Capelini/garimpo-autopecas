#!/usr/bin/env bash
# Portao de independencia (decisao 10): prova que o repositorio oficial roda sozinho,
# sem ~/garimpo-vitor, sem o ambiente "garimpo" e sem rede.
#   Fase A (COM internet): clone limpo + ambiente novo + planilha.
#   Fase B (SEM rede):     testes, scripts e smoke do app.
# Uso:  bash scripts/08_portao_independencia.sh A /caminho/BD_Proj_Integrador_5CD.xlsx
#       (desligue a rede)   bash scripts/08_portao_independencia.sh B
set -euo pipefail
DIR="${PORTAO_DIR:-$HOME/portao-garimpo}"
ENV="garimpo-portao"
REPO="https://github.com/A-Capelini/garimpo-autopecas"
BRANCH="${PORTAO_BRANCH:-integracao}"

fase_a() {
  local xlsx="${1:?informe o caminho da planilha}"
  rm -rf "$DIR"
  GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$BRANCH" "$REPO" "$DIR"
  cd "$DIR"
  conda env remove -y -n "$ENV" 2>/dev/null || true
  conda env create -y -f environment.yml -n "$ENV"
  conda run -n "$ENV" --no-capture-output pip install -e .
  mkdir -p data/raw && cp "$xlsx" data/raw/
  echo "FASE A OK. Desligue a rede e rode a fase B."
}

fase_b() {
  cd "$DIR"
  echo "== rede: deve falhar =="
  if curl -s --max-time 3 -o /dev/null https://pypi.org; then
    echo "ATENCAO: a rede ainda esta ligada"; exit 1
  fi
  echo "== independencia: nenhuma referencia ao repo do Vitor =="
  ! grep -rIn --exclude-dir=.git --exclude=08_portao_independencia.sh "garimpo-vitor" . || { echo "FALHOU"; exit 1; }
  run() { conda run -n "$ENV" --no-capture-output "$@"; }
  run which python
  run python -m pytest -q
  run python scripts/01_preparar_dados.py
  run python scripts/06_importar_catalogo.py
  echo "== app (smoke, 15 s) =="
  run streamlit run app/Home.py --server.headless true --server.port 8599 &
  local pid=$!
  sleep 15
  curl -s http://127.0.0.1:8599/_stcore/health; echo
  kill "$pid" 2>/dev/null || true
  echo "FASE B OK."
}

case "${1:-}" in
  A) shift; fase_a "$@";;
  B) fase_b;;
  *) echo "uso: $0 A <planilha.xlsx> | B"; exit 2;;
esac
