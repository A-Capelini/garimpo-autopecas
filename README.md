# Garimpo — busca híbrida offline de autopeças para carros antigos

Projeto Integrador IV — Fatec Cotia (Ciência de Dados).

Sistema 100% local (sem chave de API, sem enviar dados à internet) que encontra peças
no catálogo de 13 fornecedores (8.802 linhas) por texto + embeddings + filtros e, numa
fase seguinte, sugere adaptações e explica o resultado com um LLM local (Ollama).

## Regras de arquitetura (motor separado da interface)

| Pasta | Papel | Pode importar |
|---|---|---|
| `src/autopecas/` | **Motor**: dados, recuperação, avaliação, (depois) LLM | só bibliotecas de dados; **nunca** Streamlit |
| `app/` | **Interface** Streamlit (`Home.py` + `pages/`) | apenas o motor (`from autopecas...`) |
| `scripts/` | Orquestração fina (argparse + chamadas ao motor) | motor; sem regra de negócio |
| `tests/` | pytest — não dependem do Ollama nem da planilha | motor |

Todo recuperador implementa o contrato de `retrieval/base.py` (`fit`, `score_rows`,
`score_texts`). O recuperador híbrido do próximo passo só combina os scores dos outros.

## Estrutura

```
src/autopecas/
  config.py              caminhos, colunas, níveis (Básico/Intermediário/Avançado)
  data/                  loader.py · quality.py (GTIN, texto de busca) · levels.py
  retrieval/             base.py · lexical.py (TF-IDF) · ollama_embed.py
  evaluation/            gold.py (gabarito por GTIN) · metrics.py (Hit@k, Recall@k, MRR)
scripts/                 01_preparar_dados.py · 02_avaliar.py
docs/GUIA_OLLAMA.md      instalação e primeiros passos com o Ollama
data/raw/                coloque aqui BD_Proj_Integrador_5CD.xlsx (não versionado)
data/processed/          catalogo.csv e caches de embeddings (gerados)
resultados/              relatorio_qualidade.md e avaliacao.csv (gerados)
```

## Instalação

```bash
conda env create -f environment.yml
conda activate garimpo-autopecas     # o nome vem do campo `name:` do environment.yml
pip install -e .
pytest                               # 39 testes devem passar (12 do motor + 27 do codigo herdado)
```

Confirme que o ambiente está ativo antes do `pip`: o prompt deve mostrar `(garimpo-autopecas)` e
`which python` (Windows: `where python`) deve apontar para dentro de `envs/garimpo-autopecas`.
No VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → `garimpo-autopecas`.

## Uso

Linux / macOS:
```bash
cp /caminho/BD_Proj_Integrador_5CD.xlsx data/raw/
python scripts/01_preparar_dados.py                  # gera catalogo.csv e o relatório de qualidade
python scripts/02_avaliar.py --retriever lexical     # baseline sem IA
```

Windows (Anaconda Prompt ou PowerShell):
```bat
copy C:\caminho\BD_Proj_Integrador_5CD.xlsx data\raw\
python scripts\01_preparar_dados.py
python scripts\02_avaliar.py --retriever lexical
```

Depois de instalar o Ollama (`docs/GUIA_OLLAMA.md`; no Windows use o instalador de ollama.com/download):
```bash
ollama pull all-minilm
ollama pull bge-m3
python scripts/02_avaliar.py --retriever ollama --modelo all-minilm
python scripts/02_avaliar.py --retriever ollama --modelo bge-m3
```

Cada execução acrescenta linhas em `resultados/avaliacao.csv` (tabela pronta para o relatório).

## Problemas comuns

| Mensagem | Causa | Solução |
|---|---|---|
| `EnvironmentNameNotFound` | o nome no `conda activate` não é o do campo `name:` do `environment.yml` | ative com o nome correto (`conda env list` mostra os existentes) |
| `externally-managed-environment` no `pip` | o ambiente conda não estava ativo e o `pip` era o do sistema | ative o ambiente e confira com `which python`; **não** use `--break-system-packages` |
| `ModuleNotFoundError: autopecas` | falta `pip install -e .` (ou interpretador errado no VS Code) | rode `pip install -e .` dentro do ambiente |

## Código herdado (em consolidação)

O histórico do repositório de um integrante do grupo (`github.com/VicktzZ/garimpo`, 4 commits) foi
mesclado aqui em 06/11/2026. Esse código está na raiz: `garimpo/` (pacote), `app.py` (interface
Streamlit), `requirements.txt`, `.streamlit/` e, em `scripts/`, `preparar_dados.py`, `avaliar.py` e
`comparar_embeddings.py`. Os 27 testes dele estão em `tests/` (`test_avaliacao.py`, `test_banco.py`,
`test_busca.py`, `test_dados.py`, `test_embeddings.py`, `test_lexical.py`, `test_semantica.py`).

O motor de referência é `src/autopecas/`. O código herdado **ainda não foi consolidado** nele:
as duas estruturas convivem até a Etapa 1, e a raiz está no `pythonpath` do pytest só por causa disso.

## Dados sigilosos

O repositório é **privado** e os dados da empresa nunca vão para o Git: `data/raw/`, `data/processed/`
e `resultados/` estão no `.gitignore`. Confira com `git ls-files data resultados` (só deve listar `.gitkeep`).

## Atenção: o nível Básico não tem gabarito por GTIN

Entre os fornecedores A–D **não há GTIN válido compartilhado**, então o gabarito do nível
Básico tem 0 consultas (Intermediário: 294; Avançado: 385). Para o Básico será preciso um
conjunto manual de perguntas com as peças esperadas (30 a 50 itens).

As métricas só são comparáveis **entre recuperadores dentro do mesmo nível** — cada nível tem
um gabarito diferente.

## Próximos passos

1. Rodar os embeddings (`all-minilm`, um multilíngue leve e `bge-m3`) e escolher o recuperador.
2. Recuperador híbrido: filtros (fornecedor/preço/NCM) + busca textual + embeddings, com fusão de scores.
3. Modo Adaptações com dados internos (categoria, família de veículo, faixa de ano).
4. LLM local só para explicar o resultado; interface Streamlit em `app/`.
