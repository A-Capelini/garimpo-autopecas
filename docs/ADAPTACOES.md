# Modo Adaptações (Etapa 4)

Pergunta que responde: "tenho esta peça para o **veículo** (e **ano**) do cliente?". Nunca afirma
compatibilidade: o catálogo dos fornecedores não a garante, e a tela mostra isso em destaque.

## O que a tela mostra

Cada linha tem um **rótulo fechado** e um **motivo**:

| Rótulo | Quando |
|---|---|
| `citado na descrição do fornecedor` | a descrição cita o veículo (e a faixa de ano, se houver, cobre o ano pedido) e a peça confere |
| `sugestão a validar` | veículo da mesma *família no catálogo*; peça sem veículo na descrição; mesmo GTIN em outro fornecedor; peça relacionada (mesmo NCM útil, descrição de outra peça) |

Não existe "100% compatível". Itens cuja faixa de ano não cobre o ano pedido ficam ocultos e contados.

## Como é calculado (sem LLM, sem Ollama)

- **Veículo e ano** (`src/autopecas/veiculos.py`): extraídos da descrição por vocabulário do próprio
  catálogo e por regras de faixa (`85/90`, `78 A 82`, `74/`, `92>`, `ATE 83`, ano de 4 dígitos). Ano de
  2 dígitos ≤ 30 vira 20xx, acima vira 19xx. Cilindrada (1300), pneu (165/70), medidas e mês/ano
  (`08/99`) não viram ano. Campos derivados em memória: o SQLite e o `texto_busca` não mudam.
- **Família**: modelos que o catálogo cita juntos (≥ 5 descrições e ≥ 30% das do modelo menos citado;
  não transitivo). Não é base externa de plataformas; a camada ACES/PIES continua adiada.
- **NCM**: só serve como categoria quando não é o genérico 87089990.
- **Busca**: a peça é buscada **sem** o veículo (o nome do carro não domina o ranking) e os candidatos
  precisam cobrir a maioria das palavras da peça (abreviação por prefixo aceita).

## Ordenação por veículo na busca principal

Quando a consulta cita um veículo conhecido, o topo-50 da busca é reordenado em camadas (estável):
1 cita o veículo e a peça confere · 2 neutro (genérico, ou cita o veículo mas é outra peça) ·
3 cita só outro veículo ou o ano está fora da faixa. A mesma função roda no app e na avaliação
(`retrieval/veiculo.py`), então medir é medir o que o app mostra. A coluna **Veículo** explica a camada.

## Limites conhecidos

- Siglas de 2 letras (`CB.` = cabo) não casam na cobertura; peça relacionada com as mesmas palavras ainda passa.
- Ano é um envelope por item, sem associar cada faixa a um veículo; marcas (VW, FIAT) não contam como veículo.
- A regra foi desenhada olhando as mesmas 43 perguntas do gabarito: o ganho medido é **exploratório**
  e parte dele é por construção (a regra segue o critério com que o gabarito foi rotulado).

## Medir

```bash
python scripts/10_medir_veiculo.py              # léxico, bge-m3 e híbrida, sem e com a regra (Ollama)
python scripts/10_medir_veiculo.py --so-lexical # sem Ollama
```
