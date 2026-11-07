# Holdout da Etapa 4 (perguntas escritas sem ver o catálogo)

Objetivo: medir a regra de ordenação por veículo em perguntas que **não** foram usadas para desenhá-la
(a medição das 43 perguntas do gabarito do Básico é exploratória, porque a regra foi desenhada nelas).

## Regras (valem antes de medir)

1. As perguntas (`data/raw/holdout_perguntas.csv`, `autor` = Gemini) **não são reescritas** para casar com o
   catálogo. Erros de digitação e abreviações são propositais (tipos `erro_digitacao` e `coloquial`).
2. A regra de veículo fica **congelada** no commit/tag `etapa-4` até a medição terminar.
3. As peças esperadas vêm de **filtro de palavras-chave** (`scripts/11_sugerir_pecas_holdout.py`), nunca do
   ranking do sistema (decisão 33), e só valem depois que o **grupo marca S/N** em cada candidata.
4. Quais perguntas entram na medição depende **só da existência da peça**, nunca do desempenho do sistema.
5. Planilha de conferência, CSVs e gabaritos ficam em `resultados/` ou `data/raw/` (fora do Git) e não vão a
   serviço externo, incluindo o Gemini.

## Fluxo

```bash
# 1) gera a planilha de conferência e o relatório de existência de peças
python scripts/11_sugerir_pecas_holdout.py --nivel "Intermediário"

# 2) o grupo preenche S/N (+ revisor) na aba Candidatas de resultados/conferencia_holdout_intermediario.xlsx

# 3) valida e grava um gabarito por nível (a conferência do Intermediário gera também o do Básico)
python scripts/11_sugerir_pecas_holdout.py --fechar resultados/conferencia_holdout_intermediario.xlsx \
    --nivel "Intermediário" --gabarito-saida data/raw/holdout_gabarito_intermediario.csv
python scripts/11_sugerir_pecas_holdout.py --fechar resultados/conferencia_holdout_intermediario.xlsx \
    --nivel "Básico" --gabarito-saida data/raw/holdout_gabarito_basico.csv
```

O fechamento recusa o arquivo (e lista o que falta) se houver candidata sem S/N, sem `revisor` ou código fora
do formato. Pergunta sem nenhum S vira `sem_resposta`; `sem_resposta` do Gemini que ganhou peça fica sem tipo
(o grupo escolhe). O resultado é lido por `ler_gabarito` e conferido contra o catálogo antes de ser gravado.

## O que a triagem faz e não faz

- Casa cada termo por igualdade, plural, abreviação (`interr` = `interruptor`, `diant` = `dianteiro`) ou, se
  nada casar, por similaridade de grafia. Termo sem correspondente é ignorado e listado no relatório.
- Aplica o critério do gabarito do Básico (`docs/GABARITO_BASICO.md`): com veículo na pergunta, só o item que
  cita o veículo é `provável`; genérico ou de outro veículo, e lado/posição contrários, vão para `Descartadas`.
- **Não decide** "peça parecida mas diferente" (ex.: capa da correia dentada ≠ correia dentada) nem sinônimos
  que só o grupo conhece. Validação contra o gabarito do Básico (43 perguntas com resposta): o filtro lista
  118 das 123 peças esperadas (95,9%) e marca 89 como `provável` (72,4%). Não é gabarito.
