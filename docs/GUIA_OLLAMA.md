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
| LLM (resumo em linguagem natural) | `llama3.2:3b`, experimental e desligado por padrão | decisão 59: o resumo padrão é por regras (`autopecas.resumo`); medições na seção 9 |

## 6. Desempenho (i5-4690K, 16 GB, sem GPU)
- Indexar o catálogo inteiro com `bge-m3`: ≈10 min, uma vez. Depois, atualizar 100 itens novos
  levou ≈8 s (82 vetores novos) — medido em `resultados/medicao_atualizacao.md`.
- Consulta nova no app ≈ 30 ms de busca (+ ≈125 ms para vetorizar o texto da consulta na primeira vez).
- Pico ocasional de ≈ 3,3 s numa consulta inédita (2 ocorrências, no script e no app; 3 consultas inéditas seguintes levaram 24 a 38 ms). Causa não confirmada; acima da meta de 1 s.

## 7. Vetores e cache
Vetores ficam na tabela `embeddings_cache` do `data/catalogo.db`, chaveados por modelo + hash do
texto. Não existe mais cache `.npy`. Importar o mesmo texto de novo não gera novo vetor.

## 8. Testar a API na mão
```bash
curl http://127.0.0.1:11434/api/embed -d '{"model":"bge-m3","input":["pastilha de freio opala"]}'
```
Use `/api/embed` (a rota `/api/embeddings` é antiga).

## 9. LLM local (Etapa 5)
Medido em 07/11/2026 no PC de referência (i5-4690K, 4 núcleos, sem GPU), com o `bge-m3` carregado, 12 consultas
reais do tipo balcão e o caminho real do app (`Buscador` + `llm`; não inclui o desenho da tela do Streamlit).

**Decisão 59 (por delegação; o grupo pode vetar):** o resumo padrão do app é **por regras** (`autopecas.resumo`):
fornecedor, código, descrição na íntegra, preço e a situação do veículo que a própria busca calcula. O LLM é um
complemento **experimental, desligado por padrão**, com o `llama3.2:3b` e o prompt `c`. O `gemma3:1b` saiu do app.

| Modelo e prompt | Resultado nas 12 consultas (lista na mão, conferida na tela do app) |
|---|---|
| `gemma3:1b`, prompt `c` | 8 de 12 respostas com problema (afirma ou sugere que serve sem base, troca peça, funde preços); a guarda de texto bloqueou 0 |
| `llama3.2:3b`, prompt `c` | recusou 6 de 12; nas 6 consultas conferidas na tela, só 1 estava correta (as recusas de palheta, filtro de óleo, jogo de juntas e cabo de freio eram indevidas; errou o veículo em "retrovisor") |

Tempos do prompt `c` (modelo já carregado): `gemma3:1b` 1ª palavra 1,2 s (máx. 1,7 s), total 5,5 s (máx. 6,5 s),
19,7 tok/s; `llama3.2:3b` 1ª palavra 2,8 s (máx. 3,4 s), total 5,9 s (máx. 13,2 s), 7,5 tok/s. A frio:
`gemma3:1b` 7,8 s e `llama3.2:3b` 22,6 s até a primeira palavra (31,7 s no total, acima das metas de 10 s e 30 s),
por isso o app carrega o modelo ao ligar o resumo por IA.

Causa de fundo: a busca devolve itens que casam por palavra mas são outra peça (suporte de cabo, coxim,
guarda-pó) e os modelos pequenos não sabem dizer qual peça cada item é; a guarda só vê palavras ("compatível",
"serve") e não pega frases como "é a peça solicitada". Por isso o resumo por regras reproduz a descrição sem
parafrasear. Modelos `gemma3:4b` (lento, usa swap), `qwen3.5` e `gemma4` (tamanho) foram descartados; `lfm3`
exige Ollama mais novo que o disponível no conda-forge (0.32.15).

Para medir de novo (um modelo por vez, `ollama pull <modelo>` antes):
```bash
python scripts/12_medir_llm.py --modelos llama3.2:3b gemma3:1b --prompt c --rotulo minha_medicao
```
Saídas em `resultados/` (fora do Git; contêm descrições e preços reais do catálogo).

## 10. Problemas comuns
| Sintoma | O que fazer |
|---|---|
| App em "🔴 offline" | `ollama serve`; `curl .../api/tags`; `ollama list` deve mostrar `bge-m3` |
| `Modelo ... nao encontrado` | `ollama pull bge-m3` |
| Primeira busca lenta | modelo carregando / indexação inicial |
| Falta de memória | feche navegador e outros programas |
