# Guia rápido do Ollama (Ubuntu, CPU)

## O que é
O Ollama roda como um **servidor local** (porta `11434`) que carrega modelos de linguagem e de
embeddings **quantizados** (formato GGUF, motor llama.cpp) e os expõe por uma API HTTP.
Nosso código Python fala com ele por `requests` (ver `retrieval/ollama_embed.py`).
Baixar um modelo (`ollama pull`) precisa de internet **uma vez**; depois disso tudo roda offline.

## 1. Instalar
Confira o comando atual em https://ollama.com/download (pode mudar). Em Linux, o instalador oficial é:

```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl status ollama        # deve estar "active (running)"
ollama --version
```

## 2. Comandos essenciais
```bash
ollama pull bge-m3             # baixa um modelo
ollama list                    # modelos instalados e tamanho em disco
ollama run gemma3:1b --verbose # conversa no terminal; --verbose mostra tokens/s ao final
ollama ps                      # o que está carregado na memória agora
ollama rm <modelo>             # apaga um modelo
```
Sair do chat: `/bye`.

## 3. Regras do projeto (governança)
- Use **somente modelos locais**. Não use tags terminadas em `-cloud` e não faça login no Ollama:
  isso enviaria dados para fora e violaria o requisito de 100% offline.
- Prova de soberania para a banca: depois de baixar os modelos, **desligue a rede** e rode o sistema.

## 4. Modelos para testar neste PC (i5-4690K, 16 GB RAM, sem GPU)
Confira nomes e tags em https://ollama.com/library antes de baixar.

| Uso | Candidatos | Observação |
|---|---|---|
| Embeddings | `all-minilm` (23M, modelo em inglês) · `paraphrase-multilingual` · `embeddinggemma` · `bge-m3` (568M, multilíngue) | testar todos no gabarito; a indexação é feita uma vez |
| LLM (explicação) | `gemma3:1b` → `qwen3:1.7b` → `llama3.2:3b` → `gemma3:4b` | começar pelo menor e subir só se o português ficar ruim |

Meça a velocidade real de cada LLM com `ollama run <modelo> --verbose` (linha `eval rate`).
Modelos com modo de raciocínio (como o Qwen 3) gastam tokens "pensando"; há opção para desligar —
veja `ollama run --help`.

## 5. Expectativa de desempenho
Este processador (Haswell, 4 threads, memória DDR3) é bem mais lento que o de um PC atual.
Planeje respostas de **vários segundos** e mantenha o prompt curto (poucas linhas recuperadas,
texto enxuto). Por isso o LLM é **opcional** no projeto: a busca funciona sem ele.
Não rode outras tarefas pesadas durante as medições.

## 6. Testar a API na mão
```bash
curl http://localhost:11434/api/embed -d '{"model":"bge-m3","input":["pastilha de freio opala"]}'
curl http://localhost:11434/api/generate -d '{"model":"gemma3:1b","prompt":"Diga oi","stream":false}'
```
Use `/api/embed` (a rota `/api/embeddings` é antiga).

## 7. Onde ficam os modelos
Em instalação como serviço no Linux, normalmente em `/usr/share/ollama/.ollama/models`, na
partição do sistema. Para usar outro disco, defina `OLLAMA_MODELS` num override do serviço
(`sudo systemctl edit ollama`) e garanta que o usuário `ollama` tenha permissão na pasta.

## 8. Problemas comuns
| Sintoma | O que fazer |
|---|---|
| `Nao consegui falar com o Ollama` | `systemctl status ollama` ou `ollama serve` |
| `Modelo ... nao encontrado` | `ollama pull <modelo>` |
| Primeira chamada lenta | o modelo está carregando; `keep_alive` mantém na memória |
| Falta de memória | usar modelo menor; fechar navegador e outros programas |
