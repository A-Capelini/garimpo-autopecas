# Gabarito manual do nível Básico (Etapa 3)

No Básico (fornecedores A a D, 765 itens) não há GTIN válido compartilhado, então não existe
gabarito automático. O grupo escreve **perguntas em texto livre** e marca **quais peças
atendem** cada uma. Com isso medimos léxico, bge-m3 e híbrida no Básico e verificamos o
comportamento com texto livre (as medições anteriores usam itens do catálogo como consulta).

**Meta:** pelo menos **30 perguntas com resposta** (o critério da Etapa 3). Faixa recomendada: 30 a 50.

## Onde fica o arquivo (sigilo)

`data/raw/gabarito_basico.csv` — **fora do Git** (a pasta `data/raw/` é ignorada), porque aponta
para peças da planilha sigilosa. No repositório há só um modelo com códigos inventados:
`docs/exemplos/gabarito_basico_modelo.csv`. Os arquivos gerados na medição (`resultados/`) também
ficam fora do Git, porque trazem descrições e preços.

## Formato

CSV UTF-8, uma linha por pergunta. Delimitador `;` (o Excel em português usa esse), `,` ou tabulação.

```
id;pergunta;pecas;tipo;autor;observacao
Q01;cabo do capô do gol;C:02864;direta;Anderson;
Q02;embreagem do fiesta;A:130781|C:07624;coloquial;Marcelo;duas peças servem
Q03;pastilha de freio de moto;;sem_resposta;Victor;não existe no Básico
```

| Coluna | Obrigatória | O que vai |
|---|---|---|
| `id` | sim | Único (`Q01`, `Q02`…) |
| `pergunta` | sim | O texto como um cliente digitaria |
| `pecas` | sim | `FORNECEDOR:código` separados por `\|`. Vazio só em `sem_resposta` |
| `tipo` | não | `direta`, `coloquial`, `erro_digitacao`, `sem_veiculo`, `por_codigo` ou `sem_resposta` |
| `autor` | não | Quem escreveu |
| `observacao` | não | Livre |

**Por que `C:02864` e não só `02864`:** o Excel apaga zeros à esquerda de números (`02864` vira
`2864`). Com a letra na frente a célula continua texto. A chave `FORNECEDOR:código` é a mesma do catálogo
(`fornecedor` + `codigo_interno`). Se o código estiver errado, a validação diz qual linha e sugere o código
certo quando o problema for o zero perdido.

**Salvar no Excel:** *Salvar como → CSV UTF-8 (delimitado por vírgulas)*. O formato "CSV (separado por
vírgulas)" simples também é aceito (Windows-1252), mas UTF-8 é mais seguro para acentos.

## Como escrever (para o resultado valer)

1. **Escreva como o cliente fala, sem copiar a descrição do catálogo.** Copiar a descrição favorece a busca
   por texto e infla o resultado. Ex.: "cabo do capô do gol", não "CABO CAPO C/ALAV. GOL/SAV/PAR/VOY".
2. **Ache as peças filtrando a planilha à mão** (aba `CONSOLIDADO`, colunas A = código, F = fornecedor), **sem
   usar o Garimpo**. Quem usa o sistema para achar a resposta mede o sistema contra ele mesmo.
3. **Marque todas as peças que atendem**, de qualquer fornecedor A a D. Peça que serve e não foi marcada
   derruba a métrica sem culpa do sistema.
4. **Misture os tipos.** Sugestão (não é regra): maioria `direta` e `coloquial`; algumas com erro de
   digitação, sem o veículo, e com código do fabricante; **2 a 5 `sem_resposta`** (peças que não existem no
   Básico), que servem para ver o que o sistema devolve quando não há resposta.
5. **Divida o trabalho** (8 a 10 perguntas por pessoa) e **peça que outra pessoa confira** as peças de cada
   pergunta.
6. **Não troque perguntas depois de ver o ranking.** Ajustar o gabarito para o sistema acertar mais é
   viés. O que é permitido: **completar peças esquecidas** que a revisão mostrar (veja abaixo). Registre no
   Diário de Bordo o que mudou e por quê.

## Validar e medir

```bash
# 1) só conferir o arquivo contra o catálogo (não precisa do Ollama)
python scripts/02_avaliar.py --retriever lexical --gabarito data/raw/gabarito_basico.csv --so-validar

# 2) validar + medir léxico, bge-m3 e híbrida escolhida (precisa do Ollama no ar)
bash scripts/09_medicao_etapa3.sh
```

Também dá para medir um recuperador por vez, com os mesmos parâmetros do gabarito por GTIN
(`--retriever lexical|ollama|hibrido`, `--preset escolhido`).

A validação aponta, de uma vez só, todos os problemas (linha e código) e avisa se há menos de 30 perguntas com resposta.

## Como ler o resultado

- **Hit@k:** fração de perguntas com pelo menos uma peça esperada entre as k primeiras. **MRR:** média de
  1/posição da primeira peça esperada. **Recall@10:** média da fração das peças esperadas que aparecem nas 10 primeiras.
- **Intervalo entre colchetes (IC95%, bootstrap sobre as perguntas):** com 30 a 50 perguntas ele é largo
  (dezenas de pontos). **Diferença pequena entre léxico e híbrida, com intervalo que contém zero, não é
  conclusiva**; diga isso no relatório. A diferença pareada híbrido − léxico é impressa pelo modo `todos`.
- **Sem evidência:** peça com score 0 não aparece na lista (igual ao app) e conta como posição n_docs+1.
- **Tempos:** `ms_por_consulta` na primeira execução inclui o embedding de cada pergunta nova no Ollama; nas
  execuções seguintes os vetores das perguntas já estão em cache e o tempo cai. Não use como latência final.
- **Rastreabilidade:** cada linha de `resultados/avaliacao_texto.csv` traz `gabarito_sha256` (os 12 primeiros
  caracteres do hash do arquivo). Se o gabarito mudar, o hash muda: números com hashes diferentes não são comparáveis.

## Revisão manual (completar o gabarito)

`resultados/revisao_basico_<recuperador>.md` mostra, por pergunta, a posição das peças esperadas e o top-5
(✅ = peça do gabarito). Olhe os itens não marcados que aparecem no topo: se um **serve de verdade**,
acrescente-o em `pecas` e meça de novo. Nas `sem_resposta`, veja se o sistema devolveu algo parecido com a
peça (ele costuma devolver vizinhos; calibrar um limiar de similaridade é pendência).

## Limites (para o relatório)

- Vale só para o **Básico**: nos outros níveis os fornecedores E a N podem ter peças que atendem a pergunta e
  que o gabarito não lista.
- Com 30 a 50 perguntas escritas por poucas pessoas, o resultado mostra uma tendência, não uma garantia.
- Avaliar com o texto livre difere de usar itens como consulta: não compare números das duas medições lado a lado.
