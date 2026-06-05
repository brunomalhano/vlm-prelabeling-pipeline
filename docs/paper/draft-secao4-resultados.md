# Seção 4 — Resultados

> **Run de referência**: `run-20260515T011220Z`  
> **Dados-fonte**: `results/run-20260515T011220Z/{e1,e3,e4,e5}_*.csv`, `results/tables/stat_*.csv` (incluindo normalização macro e balanceamento por classe+instância)  
> **Extensão-alvo**: ~1.600 palavras

Esta seção apresenta os resultados dos experimentos executados sobre o conjunto de validação COCO 2017 (500 imagens, 10 classes, seed=42). Os resultados estão organizados em quatro blocos: qualidade-base por categoria (E1), impacto da formulação do prompt (E3), diagnóstico de bottleneck por estágio (E4) e taxonomia de falhas (E5). A análise estatística inclui intervalos de confiança bootstrap (10.000 reamostragens), testes de Wilcoxon pareados por imagem e tamanho de efeito de Cliff's delta.

---

## 4.1 Qualidade-Base do Pipeline (E1)

O Experimento E1 estabelece a qualidade de referência do pipeline com prompts do tipo *Simples* em inglês — o formato padrão na literatura de open-vocabulary detection.

**mIoU global (três visões complementares)**: Com prompts simples, o pipeline atinge **0,527** [IC 95%: 0,515–0,539] em micro-average (ponderado por instância), **0,506** [0,489–0,522] em macro-average (média por classe), e **0,513** [0,477–0,547] em avaliação estritamente balanceada por classe+instância (54 instâncias por classe). As três medidas convergem para a mesma leitura qualitativa e tornam explícito o efeito do desbalanceamento de classes.

| Classe | mIoU | Taxa de Detecção | N instâncias |
|--------|------|------------------|--------------|
| cat | 0,686 | 92,6% | 54 |
| dog | 0,668 | 89,6% | 67 |
| person | 0,544 | 82,9% | 1.863 |
| pizza | 0,457 | 60,0% | 150 |
| bottle | 0,429 | 68,0% | 535 |
| car | 0,422 | 57,4% | 411 |
| cup | 0,342 | 43,0% | 512 |
| chair | 0,297 | 50,0% | 722 |
| bicycle | 0,264 | 48,0% | 171 |
| apple | 0,231 | 35,2% | 105 |

*Tabela 4.1: Qualidade-base do pipeline por classe (E1, prompts simples, N = 4.590 instâncias GT).*

**Padrão observado**: Classes com alta saliência visual (cat, dog) atingem mIoU > 0,65 e taxas de detecção acima de 89%. Classes com objetos pequenos, baixo contraste ou oclusão frequente (apple, bicycle, chair) apresentam mIoU < 0,30 e taxas de detecção abaixo de 50%. A variação entre o melhor caso (cat: 0,686) e o pior (apple: 0,231) é de 3×, indicando que a qualidade do pipeline é fortemente mediada pelas propriedades visuais da categoria.

**Distribuição de utilidade** (média das 10 classes com prompts simples):

| Classificação | Percentual |
|--------------|-----------|
| Boa (≥ 0,75) | ~36% |
| Corrigível (0,50–0,75) | ~19% |
| Ruim (< 0,50) | ~13% |
| Não detectada | ~33% |

---

## 4.2 Impacto da Formulação do Prompt (E3) — Resultado Central

O Experimento E3 é o resultado central deste trabalho. Avalia as quatro formulações de prompt (Simples, Direto, Contextual, Orientado a objeto) mantendo todos os demais parâmetros constantes.

### 4.2.1 Resultado Global

**O resultado mais relevante é contraintuitivo**: o prompt *Simples* — um único nome de classe — supera todas as formulações elaboradas em todos os cenários de agregação (micro, macro e classe+instância balanceada), com significância estatística robusta nos testes pareados por imagem.

| Formulação | mIoU Micro | mIoU Macro | mIoU Balanceado | Detecção Micro | Detecção Macro | Detecção Balanceada | p (Wilcoxon) | Cliff's δ |
|-----------|------------|------------|-----------------|----------------|----------------|---------------------|--------------|-----------|
| **Simples** | **0,527** [0,515; 0,539] | **0,506** [0,489; 0,522] | **0,513** [0,477; 0,547] | **66,3%** [64,9%; 67,7%] | **62,7%** [60,7%; 64,6%] | **63,3%** [59,1%; 67,4%] | — | — |
| Contextual | 0,250 [0,239; 0,261] | 0,366 [0,349; 0,383] | 0,355 [0,319; 0,390] | 30,7% [29,4%; 32,0%] | 44,5% [42,5%; 46,5%] | 43,5% [39,4%; 47,6%] | 6,95 × 10⁻⁵⁵ | 0,43 (médio) |
| Direto | 0,210 [0,200; 0,221] | 0,325 [0,308; 0,341] | 0,348 [0,313; 0,384] | 25,4% [24,1%; 26,6%] | 38,9% [37,0%; 40,7%] | 41,5% [37,4%; 45,7%] | 1,08 × 10⁻⁶⁶ | 0,52 (grande) |
| Orientado a obj. | 0,204 [0,194; 0,214] | 0,305 [0,287; 0,321] | 0,298 [0,264; 0,333] | 24,8% [23,6%; 26,1%] | 36,8% [34,8%; 38,7%] | 35,4% [31,3%; 39,4%] | 5,40 × 10⁻⁶⁵ | 0,52 (grande) |

*Tabela 4.2: mIoU e taxa de detecção por formulação de prompt em três cenários de agregação, com análise estatística (E3, N = 4.590 instâncias GT por condição; balanceado: 54 instâncias por classe).* 

Em micro-average, o prompt *Simples* produz mIoU **2,1–2,6× superior** às formulações alternativas. Em macro-average e no cenário balanceado, o ganho permanece substancial, porém menos inflado pelo desbalanceamento de classes (tipicamente **1,4–1,7×**). Os testes de Wilcoxon rejeitam $H_0$ em todas as comparações ($p < 10^{-54}$), com tamanho de efeito médio a grande.

### 4.2.2 Efeito sobre Taxa de Detecção vs. Qualidade Condicional

A decomposição do mIoU revela que a degradação se concentra na **taxa de detecção** (micro: 66,3% para 24,8–30,7%; macro: 62,7% para 36,8–44,5%), enquanto a qualidade condicional da máscara quando o grounding é bem-sucedido permanece alta:

| Formulação | mIoU condicional (Box IoU ≥ 0,75) |
|-----------|-----------------------------------|
| Simples | 0,838 |
| Contextual | 0,856 |
| Direto | 0,860 |
| Orientado a obj. | 0,861 |

*Tabela 4.3: Mask IoU condicional a grounding bem-sucedido (Box IoU ≥ 0,75).*

Os valores condicionais são notavelmente próximos (0,838–0,861), indicando que **quando o grounding funciona, o SAM 2.1 entrega qualidade de máscara essencialmente independente da formulação**. A degradação observada no mIoU global é quase inteiramente explicada pela queda na taxa de detecção.

### 4.2.3 Interação Formulação × Classe

O efeito da formulação não é uniforme entre categorias:

| Classe | Simples | Contextual | Direto | Objeto |
|--------|---------|-----------|--------|--------|
| cat | 0,686 | 0,673 | 0,661 | 0,620 |
| dog | 0,668 | 0,643 | 0,621 | 0,611 |
| person | 0,544 | 0,207 | 0,186 | 0,198 |
| bottle | 0,429 | 0,193 | 0,136 | 0,108 |
| chair | 0,297 | 0,161 | 0,137 | 0,139 |

Classes de alta saliência (cat, dog) sofrem degradação moderada (~10%); classes de baixa saliência (person, bottle, chair) sofrem degradação catastrófica (>60%). Isso sugere que quando o sinal visual é fraco, o modelo não compensa a diluição textual introduzida por formulações complexas.

---

## 4.3 Diagnóstico de Bottleneck: Grounding vs. Segmentação (E4)

O Experimento E4 isola a contribuição de cada estágio para a qualidade final, utilizando análise condicional.

### 4.3.1 Qualidade Condicional da Máscara

| Faixa de Box IoU | Mask IoU médio | Dice médio | Boundary F1 médio |
|-----------------|----------------|-----------|-------------------|
| ≥ 0,75 (boa) | **0,849** | **0,914** | **0,728** |
| ≥ 0,50 (aceitável) | **0,828** | **0,899** | **0,717** |

*Tabela 4.4: Qualidade da máscara condicionada à qualidade do bounding box (E4).*

Quando o bounding box é preciso (IoU ≥ 0,75), o SAM 2.1 produz Mask IoU = **0,849** e Dice = **0,914** — qualidade suficiente para pré-rotulagem automática com mínima revisão humana. O bottleneck do pipeline está integralmente no grounding.

### 4.3.2 Distribuição de Erros por Estágio

| Tipo de Erro | N | Percentual |
|-------------|---|-----------|
| **Grounding miss** (sem detecção) | 11.601 | **73,8%** |
| Falso positivo | 2.545 | 16,2% |
| Máscara incompleta (box OK, máscara ruim) | 1.156 | 7,4% |
| Box incompleto | 282 | 1,8% |
| Erro de segmentação puro (box bom, máscara ruim) | ~202 | ~2,2% |

*Tabela 4.5: Distribuição de erros por estágio (E4, N = 9.304 pares matched + 2.545 FP).*

**73,8% das falhas são grounding misses** — o pipeline falha em detectar o objeto, não em segmentá-lo. Os erros de segmentação puro representam apenas ~2,2% das falhas totais.

---

## 4.4 Taxonomia de Falhas (E5)

| Código | Tipo de Falha | Frequência |
|--------|--------------|-----------|
| F4 | Objeto pequeno (< 1% da área da imagem) | 24,1% |
| F1 | Oclusão total | 18,3% |
| F6 | Aparência atípica (ângulo/iluminação) | 17,1% |
| F3 | Baixo contraste com o fundo | 15,7% |
| F5 | Aglomeração (crowd) de mesma classe | 13,6% |
| F2 | Truncamento (objeto fora do frame) | 11,2% |

*Tabela 4.6: Taxonomia de falhas de grounding (E5).*

Classes com maior taxa de falha (chair, bottle, apple) concentram F4 e F3 (objetos pequenos e baixo contraste). Classes com menor falha (cat, dog) são dominadas por F6 (ângulo atípico) e F5 (aglomeração) — padrões tratáveis com ajuste de threshold.

---

## 4.5 Síntese dos Resultados

Os quatro experimentos convergem para três achados principais:

1. **Achado 1 — Variabilidade de baseline e efeito de agregação**: O pipeline com prompts simples alcança mIoU = 0,527 [0,515–0,539] em micro-average, 0,506 [0,489–0,522] em macro-average, e 0,513 [0,477–0,547] no cenário balanceado. A diferença entre micro e macro confirma a sensibilidade do número global ao desbalanceamento de classes.

2. **Achado 2 — Efeito da formulação**: Prompts simples superam todas as formulações elaboradas em todos os cenários de avaliação (micro/macro/balanceado), com significância estatística (p < 10⁻⁵⁴, Cliff's δ médio a grande). A degradação se manifesta principalmente na taxa de detecção, não na qualidade da máscara quando o grounding é bem-sucedido.

3. **Achado 3 — Bottleneck no grounding**: 73,8% das falhas são grounding misses. Quando o box é preciso (IoU ≥ 0,75), o SAM 2.1 entrega Mask IoU = 0,849 independentemente da formulação — confirmando que o estágio de segmentação é confiável.

Esses resultados são interpretados no contexto da literatura na Seção 5.
