# Guia rápido do Ollama neste projeto

## O que é
Servidor local (porta `11434`) que roda modelos quantizados (GGUF, llama.cpp) e os expõe por HTTP.
O código fala com ele por `requests` (`retrieval/ollama_embed.py`). Baixar um modelo
(`ollama pull`) precisa de internet **uma vez**; depois tudo roda offline.

## 1. Instalar (dentro do ambiente conda do projeto)
O Ollama já vem no `environment.yml` (conda-forge, build `cpu_*`). Não instale por `snap` nem
por `curl | sh`.
```bash
conda activate garimpo-autopecas
which ollama                     # deve estar dentro de envs/garimpo-autopecas
ollama --version
```
O `environment.yml` define `OLLAMA_NO_CLOUD=1`, que desliga os recursos de nuvem.

## 2. Subir e conferir o servidor
```bash
ollama serve 2>&1 | tee /tmp/ollama_serve.log      # terminal dedicado; Ctrl+C para parar
curl -s http://127.0.0.1:11434/api/tags            # lista os modelos instalados
pkill -f "ollama serve"                            # parar de outro terminal
```
`address already in use` significa que já há um servidor rodando — é só usá-lo.

## 3. Comandos essenciais
```bash
ollama pull bge-m3     # baixa o modelo de embeddings (internet, uma vez)
ollama list            # modelos e tamanho em disco
ollama ps              # o que está na memória agora
ollama rm <modelo>     # apaga
```

## 4. Regras do projeto
- Somente modelos locais. Nada de tags `-cloud` e nada de `ollama signin`/login.
- Prova de soberania: depois de baixar os modelos, **desligue a rede** e rode o app.

## 5. Modelos
| Uso | Modelo | Situação |
|---|---|---|
| Embeddings | `bge-m3` (568M, multilíngue) | decidido; vetores guardados em `data/catalogo.db` |
| LLM (explicação) | candidatos `gemma3:1b`, `qwen3:1.7b`, `llama3.2:3b`, `gemma3:4b` | Etapa 5, ainda não escolhido; confira nomes em https://ollama.com/library |

## 6. Desempenho (i5-4690K, 16 GB, sem GPU)
- Indexar o catálogo inteiro com `bge-m3`: ≈10 min, uma vez. Depois, atualizar 100 itens novos
  levou ≈8 s (82 vetores novos) — medido em `resultados/medicao_atualizacao.md`.
- Consulta nova no app ≈ 30 ms de busca (+ ≈125 ms para vetorizar o texto da consulta na primeira vez).

## 7. Vetores e cache
Vetores ficam na tabela `embeddings_cache` do `data/catalogo.db`, chaveados por modelo + hash do
texto. Não existe mais cache `.npy`. Importar o mesmo texto de novo não gera novo vetor.

## 8. Testar a API na mão
```bash
curl http://127.0.0.1:11434/api/embed -d '{"model":"bge-m3","input":["pastilha de freio opala"]}'
```
Use `/api/embed` (a rota `/api/embeddings` é antiga).

## 9. Problemas comuns
| Sintoma | O que fazer |
|---|---|
| App em "🔴 offline" | `ollama serve`; `curl .../api/tags`; `ollama list` deve mostrar `bge-m3` |
| `Modelo ... nao encontrado` | `ollama pull bge-m3` |
| Primeira busca lenta | modelo carregando / indexação inicial |
| Falta de memória | feche navegador e outros programas |
