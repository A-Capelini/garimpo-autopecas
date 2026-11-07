# Garimpo — busca híbrida offline de autopeças para carros antigos

Projeto Integrador IV — Fatec Cotia (Ciência de Dados, Grupo 2).

Sistema 100% local (sem chave de API, sem enviar dados à internet) que encontra peças no
catálogo de 13 fornecedores (8.802 linhas) combinando **busca textual (TF-IDF)**, **embeddings
(bge-m3 via Ollama)** e **filtros**. Se o Ollama estiver parado, o app cai para a busca textual e
avisa na tela. Adaptações por veículo e explicação por LLM local ficam para as Etapas 4 e 5.

## Arquitetura (motor separado da interface)

| Pasta | Papel | Pode importar |
|---|---|---|
| `src/autopecas/` | **Motor**: dados, recuperação, catálogo SQLite, busca, avaliação | só bibliotecas de dados; **nunca** Streamlit |
| `app/` | **Interface** Streamlit (`Home.py` + `pages/`) | apenas o motor (`from autopecas...`) |
| `scripts/` | Orquestração fina (argparse + chamadas ao motor) | motor; sem regra de negócio |
| `tests/` | pytest — não dependem do Ollama nem da planilha | motor |

```
src/autopecas/
  config.py              caminhos, colunas, níveis, CATALOGO_DB
  data/                  loader.py · quality.py (GTIN, texto de busca) · levels.py
  retrieval/             lexical.py (TF-IDF) · ollama_embed.py (bge-m3 + cache) · hybrid.py (RRF)
  store/                 catalogo.py (SQLite, upsert, vetores) · medicao.py (custo de atualizar)
  search.py              Buscador: filtros, híbrida/lexical, equivalentes por GTIN, ordenação por veículo
  veiculos.py            extração de veículo e ano (campos derivados em memória) · família pelo catálogo
  adaptacoes.py          modo Adaptações: blocos rotulados, sempre "sugestão a validar"
  retrieval/veiculo.py   regra de camadas por veículo (mesma função no app e na avaliação)
  evaluation/            gold.py (gabarito por GTIN) · metrics.py (Hit@k, Recall@k, MRR, IC95%)
                         gabarito_texto.py + runner_texto.py (gabarito manual em texto livre)
app/Home.py              busca, filtros, status do Ollama, equivalentes por GTIN
app/pages/1_Atualizar_catalogo.py   envio da planilha, prévia e aplicação
app/pages/2_Adaptacoes.py           peça + veículo (+ ano): citado, família, genérico; sempre a validar
scripts/                 01 preparar · 02 avaliar · 03 ablação · 04 calibrar · 06 importar · 07 medir · 09 gabarito · 10 veículo
docs/GUIA_OLLAMA.md      Ollama neste projeto
docs/GABARITO_BASICO.md  como o grupo escreve e valida o gabarito manual do Básico
docs/ADAPTACOES.md       modo Adaptações: regras, rótulos, limites e como medir
data/raw/                coloque aqui BD_Proj_Integrador_5CD.xlsx (não versionado)
data/processed/          catalogo.csv (gerado)
data/catalogo.db         catálogo SQLite + vetores (gerado, não versionado)
resultados/              relatórios e medições (gerados, não versionados)
```

Parâmetros da busca híbrida (decisões do projeto): fusão RRF com peso 3:1 (léxico:semântico),
`k_rrf=1`, profundidade 50, modelo `bge-m3`. Filtros são aplicados **antes** do top-k.

## Instalação

```bash
conda env create -f environment.yml
conda activate garimpo-autopecas     # nome vem do campo `name:` do environment.yml
which python                         # deve apontar para dentro de envs/garimpo-autopecas
pip install -e .
pytest                               # 132 testes (os do app exigem Streamlit, já no ambiente)
```

O `environment.yml` é a fonte única das dependências, inclusive o Ollama (conda-forge) e
`OLLAMA_NO_CLOUD=1`. Não use `pip --break-system-packages`. No VS Code: *Python: Select
Interpreter* → `garimpo-autopecas`.

## Uso

1. Ollama (detalhes em `docs/GUIA_OLLAMA.md`), em um terminal separado, com o ambiente ativo:
   ```bash
   ollama serve
   ollama pull bge-m3          # só na primeira vez (precisa de internet)
   ```
2. Dados (a planilha é sigilosa e nunca vai para o Git):
   ```bash
   cp /caminho/BD_Proj_Integrador_5CD.xlsx data/raw/
   python scripts/01_preparar_dados.py                    # catalogo.csv + relatório de qualidade
   python scripts/06_importar_catalogo.py                 # cria data/catalogo.db
   ```
3. App:
   ```bash
   streamlit run app/Home.py
   ```
   A primeira busca híbrida indexa os itens ainda sem vetor (≈10 min no PC de desenvolvimento,
   uma única vez); depois os vetores ficam no `catalogo.db`.

Atualizar o catálogo: página **Atualizar catálogo** do app (prévia antes de aplicar) ou
`python scripts/06_importar_catalogo.py --xlsx <arquivo> [--sincronizar]`. Só os textos novos
são vetorizados. `scripts/07_medir_atualizacao.py` mede o custo (relatório em
`resultados/medicao_atualizacao.md`).

Avaliação (baseline e comparações): `python scripts/02_avaliar.py --retriever lexical`
(ou `--retriever ollama --modelo bge-m3`). Cada execução acrescenta linhas em
`resultados/avaliacao.csv`. A configuração escolhida (decisões 21 a 23) é `--preset escolhido`.

Gabarito manual do Básico (perguntas em texto livre; veja `docs/GABARITO_BASICO.md`):
```bash
python scripts/02_avaliar.py --retriever lexical --gabarito data/raw/gabarito_basico.csv --so-validar
bash scripts/09_medicao_etapa3.sh      # léxico, bge-m3 e híbrida; resultados/avaliacao_texto.csv
python scripts/10_medir_veiculo.py     # Etapa 4: base x +veiculo (precisa do Ollama; --so-lexical não precisa)
```

## Problemas comuns

| Mensagem / sintoma | Causa | Solução |
|---|---|---|
| `EnvironmentNameNotFound` | nome errado no `conda activate` | `conda env list` |
| `externally-managed-environment` no `pip` | ambiente conda inativo, `pip` do sistema | ative o ambiente; confira `which python`; **não** use `--break-system-packages` |
| `ModuleNotFoundError: autopecas` | falta `pip install -e .` | rode dentro do ambiente |
| App mostra "🔴 offline — usando só lexical" | Ollama parado ou sem o `bge-m3` | `ollama serve`; `ollama list` |
| `address already in use` no `ollama serve` | já há um servidor rodando | `curl -s http://127.0.0.1:11434/api/tags` |

## Dados sigilosos

Os dados da empresa nunca vão para o Git: `data/raw/`, `data/processed/`, `resultados/`, `*.xlsx`
e `*.db` estão no `.gitignore`. Confira com `git ls-files data resultados` (só `.gitkeep`).
O repositório está **público até o fim do projeto** e será tornado privado depois.

## Atenção: o nível Básico não tem gabarito por GTIN

Entre os fornecedores A–D não há GTIN válido compartilhado: o gabarito Básico tem 0 consultas
(Intermediário: 294; Avançado: 385). O Básico usa um **gabarito manual** (30 a 50 perguntas em texto
livre, feito pelo grupo; `docs/GABARITO_BASICO.md`), que fica fora do Git. As métricas só são
comparáveis entre recuperadores dentro do mesmo nível e do mesmo tipo de gabarito (itens como
consulta × perguntas em texto livre).

## Independência do repositório

O código que veio do repositório de um integrante (`garimpo/`, `app.py`, `requirements.txt` e três
scripts) foi removido em 06/11/2026, depois que o **portão de independência** passou: clone limpo,
ambiente novo, rede desligada, sem o repositório de origem (`scripts/08_portao_independencia.sh`,
fases A e B). O histórico dele continua no Git.

## Próximos passos

Etapa 3: gabarito Básico manual (código pronto; falta o grupo escrever as perguntas) · Etapa 4: modo Adaptações (código pronto; medição híbrida/bge-m3 depende do Ollama) · Etapa 5: LLM local
para explicar o resultado · Etapa 6: avaliação final e relatório. P2: 18/11/2026.
