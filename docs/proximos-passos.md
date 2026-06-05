# Próximos Passos — Pipeline VLM de Pré-Rotulagem

> **Criado em**: 2026-05-29  
> **Status**: Planejado (não iniciado)  
> **Run atual de referência**: `run-20260515T011220Z` (500 imagens, 4.590 instâncias GT, seed=42)

---

## 1. Balanceamento de Instâncias por Classe

### 1.1 Problema Identificado

O mIoU global de 0,527 (prompts simples) é calculado sobre 4.590 instâncias GT distribuídas de forma **altamente desbalanceada** entre as 10 classes:

| Classe | N instâncias | % do total | mIoU (simples) |
|--------|-------------|-----------|----------------|
| person | 1.863 | 40,6% | 0,668 |
| chair | 722 | 15,7% | 0,337 |
| bottle | 535 | 11,7% | 0,558 |
| cup | 512 | 11,2% | 0,372 |
| car | 411 | 9,0% | 0,476 |
| bicycle | 171 | 3,7% | 0,303 |
| pizza | 150 | 3,3% | 0,490 |
| apple | 105 | 2,3% | 0,260 |
| dog | 67 | 1,5% | 0,767 |
| cat | 54 | 1,2% | 0,826 |

**Ratio max/min**: 34,5× (person: 1.863 vs cat: 54).

**Risco**: O mIoU global é dominado por `person` (40,6% das instâncias), que tem mIoU = 0,668. Classes com poucas instâncias mas alto mIoU (cat: 0,826, dog: 0,767) têm peso negligível na média. Uma mudança na distribuição de classes alteraria o mIoU global sem mudança real na qualidade do pipeline — isso é uma brecha metodológica que um revisor pode questionar.

### 1.2 Opções de Mitigação

#### Opção A — Macro-Average (média das médias por classe)

Calcular o mIoU por classe primeiro, depois fazer a média não-ponderada das 10 classes. Cada classe contribui igualmente para o resultado global, independente do número de instâncias.

$$\text{Macro-mIoU} = \frac{1}{K} \sum_{k=1}^{K} \text{mIoU}_k$$

**Prós**: Simples de implementar (alteração apenas no cálculo, sem re-execução do pipeline). Protocolo padrão em benchmarks de segmentação semântica.  
**Contras**: Não resolve o problema de classes com N pequeno (cat: 54, dog: 67) terem intervalos de confiança largos. Não altera os dados brutos.

**Macro-mIoU estimado** (média simples dos 10 valores da tabela): ~0,406 — significativamente diferente do micro-average atual de 0,527, o que reforça a importância desta decisão.

#### Opção B — Subamostragem Estratificada com Cap por Classe

Re-amostrar as instâncias GT para que cada classe tenha no máximo N_max instâncias (e.g., N_max = 100 ou N_max = min(N_classes) = 54). Para classes com mais instâncias, amostrar aleatoriamente (seed fixa). Para classes com menos, manter todas.

**Prós**: Equaliza a contribuição de cada classe. Reduz o viés de `person`.  
**Contras**: Descarta dados (1.863 → 54 para person = 97% descartado). Pode reduzir o poder estatístico para testes pareados por imagem. **Requer re-execução da análise estatística** mas NÃO do pipeline (os dados brutos já existem).

#### Opção C — Ponderação Inversa (Inverse Frequency Weighting)

Ponderar cada instância pelo inverso da frequência da sua classe: $w_k = \frac{N_{total}}{K \cdot N_k}$.

**Prós**: Usa todos os dados. Equaliza a contribuição efetiva por classe.  
**Contras**: Instâncias de classes raras (cat, dog) recebem peso ~34× maior, amplificando ruído individual.

#### Opção D — Reportar Ambos (Micro + Macro) — **Recomendada**

Reportar tanto o micro-average (ponderado por instância, atual 0,527) quanto o macro-average (média por classe, ~0,406) no paper. Isso é prática padrão na literatura de segmentação e deixa transparente o efeito do desbalanceamento.

**Ação mínima**: Adicionar uma coluna de macro-average nas tabelas e uma nota metodológica no paper explicando a diferença. **Não requer re-execução do pipeline.**

### 1.3 Decisão Pendente

- [ ] Escolher entre Opções A/B/C/D
- [ ] Se Opção B: definir N_max (54? 100? 150?)
- [ ] Se Opção D: calcular macro-averages e atualizar tabelas do paper
- [ ] Avaliar se o bootstrap precisa ser refeito com a nova estratégia

### 1.4 Impacto nos Resultados Atuais

Independente da opção escolhida, as **conclusões qualitativas não mudam**:
- Simples > Elaborado (o efeito é consistente em todas as 10 classes individualmente)
- Bottleneck no grounding (73,8% das falhas são misses — isso é estrutural, não afetado pelo balanceamento)
- SAM 2.1 é confiável (Mask IoU ≥ 0,849 quando box bom — métrica condicional, não afetada pelo balanceamento)

O que muda é o **valor absoluto do mIoU global** — que é o número mais citado no paper.

---

## 2. Re-Execução do Pipeline

### 2.1 Pré-Condições

Antes de re-executar, resolver o Passo 1 (balanceamento). Se a decisão for:
- **Opção A ou D**: Não é necessário re-executar o pipeline. Apenas recalcular as métricas agregadas.
- **Opção B**: Re-executar apenas a análise estatística (não o pipeline de inferência).
- Se houver **qualquer alteração no código do pipeline** (e.g., novo threshold, nova lógica de matching): re-executar tudo.

### 2.2 Checklist de Re-Execução

Se necessário re-executar o pipeline completo:

- [ ] Verificar que o código do pipeline está commitado e versionado
- [ ] Confirmar ambiente: Azure Container Apps, GPU A100 (40 GB), mesma imagem Docker
- [ ] Manter seed=42 para amostragem estratificada
- [ ] Gerar novo run ID: `run-YYYYMMDDTHHMMSSZ`
- [ ] Executar os 4 experimentos (E1, E3, E4, E5) — E2 removido
- [ ] Coletar CSVs: `e1_overall.csv`, `e3_prompt_type.csv`, `e4_conditional.csv`, `e4_error_source.csv`, `e5_failure_taxonomy.csv`
- [ ] Re-executar `statistical_analysis.py` sobre os novos dados brutos
- [ ] Comparar resultados novos vs. `run-20260515T011220Z` para verificar reprodutibilidade
- [ ] Atualizar todas as tabelas do paper com os novos valores

### 2.3 Cenário de Re-Execução Parcial (Apenas Estatística)

Se o pipeline não precisar ser re-executado (Opções A/D):

- [ ] Modificar `statistical_analysis.py` para reportar macro-average além de micro-average
- [ ] Recalcular bootstrap CIs para macro-average
- [ ] Verificar se testes de Wilcoxon precisam de ajuste (são pareados por imagem, não por instância — possivelmente sem mudança)
- [ ] Gerar novos `stat_*.csv` com colunas adicionais
- [ ] Atualizar tabelas do paper

---

## 3. Validação da Literatura sobre Métricas e Critérios de IoU

### 3.1 Objetivo

Verificar se as métricas (mIoU, Dice, Boundary F1) e os critérios de avaliação (faixas Boa/Corrigível/Ruim) utilizados no paper estão alinhados com a literatura para este tipo de problema (pré-rotulagem assistida por VLMs em segmentação de instâncias).

### 3.2 Questões a Investigar

1. **mIoU como métrica primária**: O mIoU é a métrica padrão para segmentação semântica (Everingham et al., 2010), mas para **segmentação de instâncias** o COCO usa AP (Average Precision) em múltiplos limiares de IoU. O paper usa mIoU porque avalia qualidade de pré-rótulos individuais, não precisão de detecção — mas essa escolha precisa ser melhor justificada.

2. **Limiares de utilidade (0,50 e 0,75)**: Os limiares de IoU ≥ 0,75 (Boa), 0,50–0,75 (Corrigível), < 0,50 (Ruim) são definidos como proxy operacional. Verificar se existem estudos empíricos que validam esses limiares com medições de tempo de anotação (e.g., quanto tempo um anotador leva para corrigir uma máscara com IoU = 0,60 vs. criar do zero?).

3. **Dice vs. IoU**: O Dice Score é amplamente usado em segmentação médica. Verificar se há preferência na literatura de anotação assistida por uma métrica sobre a outra, e se a relação Dice = 2·IoU/(1+IoU) é suficiente para não reportar ambas.

4. **Boundary F1**: Utilizado para avaliar qualidade de bordas. Verificar se a tolerância de 2 pixels é padrão e se há referências para esse valor.

5. **Protocolo de avaliação COCO (AP)**: Investigar se faz sentido incluir AP@[0.50:0.95] como métrica complementar para comparação com a literatura de detecção open-vocabulary. Isso adicionaria custo de implementação mas facilitaria comparação.

### 3.3 Referências a Consultar

- [ ] Lin et al. (2014) — COCO dataset e protocolo de avaliação oficial
- [ ] Everingham et al. (2010) — VOC challenge e definição de mIoU/AP
- [ ] Kirillov et al. (2023) — SAM paper e métricas de avaliação
- [ ] Mikulová et al. (2022) — Tempo de anotação com pré-rótulos (se citado no paper)
- [ ] Cordts et al. (2016) — Cityscapes e métricas de segmentação urbana
- [ ] Surveying Annotation Time Studies — buscar meta-análise sobre IoU vs. tempo de correção
- [ ] Cheng et al. (2021) — Mask2Former e métricas de panoptic/instance segmentation

### 3.4 Possíveis Ajustes no Paper

Dependendo dos achados da revisão literária:

- [ ] Adicionar justificativa explícita para uso de mIoU (não AP) como métrica primária
- [ ] Adicionar AP@50 e AP@75 como métricas complementares (se viável sem re-execução)
- [ ] Fundamentar os limiares de utilidade com referências (ou reforçar que são proxy operacional sem validação empírica — já feito parcialmente na nota da Seção 3.10)
- [ ] Verificar se Boundary F1 com tolerância de 2px é padrão ou se deve ser parametrizado

---

## Sequência Recomendada

```
Passo 1: Decidir estratégia de balanceamento (Seção 1.3)
    ↓
Passo 2: Revisão de literatura sobre métricas (Seção 3) — pode ser feito em paralelo
    ↓
Passo 3: Implementar mudanças no cálculo/pipeline (Seção 2)
    ↓
Passo 4: Atualizar tabelas do paper (Seções 3-6) com novos valores
    ↓
Passo 5: Revisar texto do paper para consistência
```

Os Passos 1 e 2 podem ser executados em paralelo. O Passo 3 depende de ambos.
