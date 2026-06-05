# Pré-Projeto de Pesquisa

**Título**: *Less is More*: Impacto da Formulação de Prompts na Qualidade de Pré-Rotulagem com Grounding DINO e SAM 2.1

**Autor**: Bruno Malhano  
**Orientador**: *(a definir)*  
**Programa**: Tópicos Especiais em Deep Learning  
**Data**: Maio de 2026

---

## 1. Introdução e Contextualização

O desenvolvimento de sistemas de visão computacional depende fundamentalmente da disponibilidade de datasets rotulados de alta qualidade. Em tarefas de segmentação — semântica, por instância ou panóptica — a anotação requer a delimitação precisa de cada objeto ou região em nível de pixel, o que torna o processo substancialmente mais caro, demorado e suscetível a inconsistências entre anotadores (Lin et al., 2014; Infomineo, 2024). Estudos de *inter-annotator agreement* em segmentação indicam que diferentes especialistas podem divergir significativamente sobre fronteiras e categorias, e que o rótulo final frequentemente representa um consenso ponderado, não uma verdade absoluta (Yang et al., 2020).

Nesse contexto, os modelos visão-linguagem (VLMs) fundacionais abriram uma nova perspectiva para pré-rotulagem automática. O Grounding DINO (Liu et al., 2023) localiza objetos arbitrários em imagens a partir de descrições textuais, enquanto o Segment Anything Model 2.1 (Ravi et al., 2024) gera máscaras de segmentação a partir de prompts visuais como bounding boxes. A combinação desses modelos em um pipeline modular — grounding textual seguido de segmentação promptável — possibilita a geração automática de máscaras candidatas para qualquer classe descritível em linguagem natural, sem retreinamento.

Entretanto, há uma decisão prática que todo praticante deve tomar ao implantar esses pipelines e que permanece sem resposta empírica na literatura: **qual formulação de prompt usar?** A literatura de *open-vocabulary detection* emprega sistematicamente nomes de classe simples — "dog", "car", "bottle" — como prompts para o Grounding DINO. Essa escolha raramente é justificada empiricamente; ela é herdada das convenções de benchmarking, não derivada de evidência experimental.

A intuição contrária é plausível: prompts mais descritivos deveriam fornecer contexto semântico adicional ao modelo. Se essa intuição estiver correta, pipelines de pré-rotulagem industriais estariam deixando desempenho na mesa. Se estiver errada — e prompts complexos degradam a qualidade — os praticantes precisam saber por quê.

---

## 2. Problema de Pesquisa

A formulação do prompt textual afeta significativamente a qualidade da pré-rotulagem gerada por um pipeline modular Grounding DINO + SAM 2.1? Mais especificamente:

1. O efeito se concentra na etapa de localização textual (*grounding*: text-to-box) ou na geração da máscara (*segmentação*: box-to-mask)?
2. Qual é o principal modo de falha do pipeline — erros de detecção ou erros de segmentação?

---

## 3. Objetivos

### 3.1 Objetivo Geral

Avaliar sistematicamente o impacto de quatro formulações de prompt textual na qualidade de máscaras de segmentação geradas por um pipeline modular Grounding DINO (Swin-B) + SAM 2.1 (Hiera Large), aplicado ao COCO val2017.

### 3.2 Objetivos Específicos

1. Estabelecer a qualidade-base (*baseline*) do pipeline por categoria de objeto utilizando prompts simples.
2. Comparar quatro formulações de prompt (simples, direto, contextual, orientado a objeto) e quantificar o impacto de cada uma nas métricas de segmentação.
3. Diagnosticar a origem dos erros por meio de análise condicional estágio-a-estágio (grounding vs. segmentação).
4. Construir uma taxonomia de falhas que identifique os padrões de erro dominantes.
5. Derivar um protocolo de avaliação em 3 camadas reutilizável para pipelines de pré-rotulagem baseados em VLMs.

---

## 4. Hipóteses

As hipóteses abaixo foram formalizadas *a priori*, antes da execução experimental. Sua verificação é apresentada na Seção 9.2 (Resultados → Verificação das Hipóteses).

| ID | Hipótese |
|----|----------|
| **H1** | Prompts simples (*single-word*, e.g. `dog`) produzem mIoU significativamente superior a formulações complexas (e.g. `the dog in the image`, `segment the dog`, `object: dog`). |
| **H2** | O efeito da complexidade do prompt concentra-se na etapa de grounding (Box IoU), não na etapa de segmentação (Mask IoU condicionado a Box IoU ≥ 0.50). |
| **H3** | A principal fonte de falhas no pipeline é o *grounding miss* (ausência de detecção), não erros de qualidade de máscara. |

---

## 5. Justificativa

A contribuição deste trabalho justifica-se em três dimensões:

**Relevância prática**: A escolha do formato de prompt afeta todos os objetos, todas as imagens e todas as iterações de anotação em pipelines de pré-rotulagem. Erros sistemáticos de formulação se propagam para a qualidade do dataset final. Equipes que implantam esses pipelines tomam essa decisão sem fundamentação empírica.

**Lacuna na literatura**: A literatura de engenharia de prompts para VLMs estabeleceu que a formulação importa para classificação (CLIP — Radford et al., 2021) e que prompts contextuais com âncora espacial ajudam em referência (APE — Zhu et al., 2023). Porém, há pouca evidência sistemática sobre o impacto de diferentes formulações de prompt em um pipeline de grounding + segmentação para pré-rotulagem — cenário de implantação mais comum para Grounding DINO + SAM. Os trabalhos existentes tipicamente adotam nomes de classe simples por convenção de benchmarking, sem justificar empiricamente essa escolha.

**Resultado contraintuitivo**: Os resultados obtidos invertem a expectativa de que descrições mais ricas melhoram o desempenho, produzindo orientação de implantação diretamente acionável.

---

## 6. Revisão da Literatura

A pesquisa situa-se na interseção de cinco eixos conceituais:

### 6.1 Segmentação Open-Vocabulary

Métodos que segmentam a partir de descrições textuais, incluindo classes não vistas no treino. Liang et al. (2023) propuseram o OVSeg, demonstrando que o CLIP sofre degradação ao processar regiões mascaradas (20,1% mIoU no ADE20K-150). Zhang et al. (2023) apresentaram um framework unificado para segmentação e detecção open-vocabulary. Zhou et al. (2025) mapearam a evolução do campo para detecção em imagens aéreas.

### 6.2 Grounding Textual

O Grounding DINO (Liu et al., 2023) combina um detector Transformer (DINO) com pré-treinamento grounded, aceitando descrições em linguagem natural para localizar objetos. A arquitetura utiliza um *dual-encoder* (Swin-B para imagem + BERT-base para texto) com atenção cruzada bidirecional em seis camadas de *feature enhancer*, seleção de queries guiada por linguagem e representação textual *sub-sentença* para evitar interferência entre categorias.

### 6.3 Segmentação Promptável

O SAM 2.1 (Ravi et al., 2024), com arquitetura Hiera Large, gera segmentações a partir de prompts visuais. A qualidade da máscara depende diretamente da qualidade do prompt visual fornecido — em pipelines modulares, a bounding box produzida pelo grounding. Estudos comparativos (SPIE, 2025) indicam que o SAM 2 supera o SAM original em tarefas especializadas.

### 6.4 Anotação Assistida e Pré-Rotulagem

Mikulová et al. (2022) demonstraram que pré-anotações de alta qualidade reduzem o esforço de revisão humana sem perda de qualidade final. Ganguly et al. (2025) propuseram o Labeling Copilot, um sistema agêntico que orquestra múltiplos modelos para anotação automatizada, alcançando 37,1% mAP no COCO. O presente trabalho avalia um pipeline *simples* (dois modelos, sem consenso) como mecanismo de pré-rotulagem para validação humana.

### 6.5 Engenharia de Prompts para VLMs

Radford et al. (2021) demonstraram que templates como "a photo of a {class}" superam nomes isolados em classificação zero-shot com CLIP (+3,5% em ImageNet). CoOp (Zhou et al., 2022) e ProDA (Lu et al., 2022) estendem esse paradigma com aprendizado de prompts. Para tarefas de grounding, a literatura é escassa — o Grounding DINO utiliza como convenção a concatenação de nomes de classe separados por ponto, sem avaliar sistematicamente outros formatos. **Esta é a lacuna que o presente trabalho endereça.**

### 6.6 Matriz Conceitual

| # | Fonte | Ano | A | B | C | D | E |
|---|-------|-----|:-:|:-:|:-:|:-:|:-:|
| 1 | Liang et al. — OVSeg (Mask-adapted CLIP) | 2023 | ✓ | | | | |
| 2 | Liu et al. — Grounding DINO | 2023 | | ✓ | | | |
| 3 | Kirillov et al. — SAM | 2023 | | | ✓ | | |
| 4 | Ravi et al. — SAM 2 | 2024 | | | ✓ | | |
| 5 | Lin et al. — Microsoft COCO | 2014 | | | | | |
| 6 | Zhang et al. — OV Seg & Det Framework | 2023 | ✓ | ✓ | | | |
| 7 | Zhou et al. — OV Detection in UAV Imagery | 2025 | ✓ | ✓ | | | |
| 8 | Ganguly et al. — Labeling Copilot | 2025 | | ✓ | ✓ | ✓ | |
| 9 | Mikulová et al. — Pre-annotation Bias | 2022 | | | | ✓ | |
| 10 | Radford et al. — CLIP | 2021 | | | | | ✓ |
| 11 | Zhou et al. — CoOp | 2022 | | | | | ✓ |
| 12 | COCONut — Modernizing COCO | 2024 | | | | ✓ | |
| 13 | Sultan et al. — GeoSAM | 2025 | | ✓ | ✓ | | |
| 14 | Infomineo — Data Annotation | 2024 | | | | ✓ | |
| 15 | Cao et al. — MAPLM | 2024 | | ✓ | | | |

**Legenda**: A = Segmentação open-vocabulary · B = Grounding textual · C = Segmentação promptável · D = Anotação assistida · E = Engenharia de prompts para VLMs

---

## 7. Metodologia

### 7.1 Classificação da Pesquisa

- **Natureza**: Aplicada e experimental
- **Abordagem**: Predominantemente quantitativa, com componente qualitativo complementar (taxonomia de falhas)
- **Procedimento**: Experimental, com variável independente manipulada (formulação do prompt)

### 7.2 Dataset

- **Fonte**: COCO val2017 — 5.000 imagens com anotações oficiais de segmentação por instância (Lin et al., 2014)
- **Amostra**: 500 imagens por amostragem estratificada (seed=42), com mínimo de 30 instâncias por classe
- **Total de instâncias avaliadas**: 4.590 por condição de prompt (18.360 instâncias no total, 4 formulações)

### 7.3 Classes Selecionadas

| Grupo Visual | Classe | Justificativa |
|-------------|--------|---------------|
| Pessoas | person | Classe mais frequente no COCO; alta variabilidade |
| Animais | dog, cat | Formas orgânicas; bordas irregulares |
| Veículos | car, bicycle | Formas rígidas; bicycle desafiante (estrutura vazada) |
| Domésticos | chair, bottle, cup | Variabilidade de formas; frequente oclusão |
| Alimentos | apple, pizza | Objetos pequenos; semântica simples |

### 7.4 Pipeline Experimental

O pipeline de pré-rotulagem é composto por duas etapas sequenciais:

$$\text{Imagem} + \text{Prompt} \xrightarrow{\text{GDINO (Swin-B)}} \text{Bounding Boxes} \xrightarrow{\text{SAM 2.1 (Hiera-L)}} \text{Máscaras} \xrightarrow{\text{Hungarian Matching}} \text{Métricas}$$

**Etapa 1 — Localização textual (Grounding DINO, Swin-B):** Dado um prompt textual $p$ e uma imagem $I$, retorna bounding boxes candidatas $B = \{b_1, ..., b_n\}$ com scores de confiança $s_i \in [0, 1]$.

**Etapa 2 — Geração de máscara (SAM 2.1, Hiera Large):** Cada bounding box aceita é fornecida como prompt visual ao SAM 2.1, que gera a máscara de segmentação correspondente.

### 7.5 Variável Independente: Formulação do Prompt

| Tipo | Exemplo | Rationale |
|------|---------|-----------|
| **Simples** | `dog` | Formato padrão na literatura (*baseline*) |
| **Direto** | `segment the dog` | Verbo de instrução explícito |
| **Contextual** | `the dog in the image` | Contexto visual explícito |
| **Orientado a objeto** | `object: dog` | Prefixo de tipo estruturado |

### 7.6 Variáveis do Estudo

| Variável | Tipo | Valores |
|----------|------|---------|
| Formulação do prompt | Independente (primária) | Simples, direto, contextual, orientado a objeto |
| Classe COCO | Moderadora | 10 classes selecionadas |
| Box IoU | Dependente (etapa 1) | Contínua [0, 1] |
| Mask IoU / mIoU | Dependente (etapa 2) | Contínua [0, 1] |
| Dice Score | Dependente (etapa 2) | Contínua [0, 1] |
| Boundary F1 | Dependente (etapa 2) | Contínua [0, 1] |
| Utilidade prática | Dependente (derivada) | Boa (≥ 0,75), Corrigível [0,50–0,75), Ruim (< 0,50) |

### 7.7 Configuração Experimental

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `box_threshold` | 0,30 | Equilíbrio precision/recall |
| `text_threshold` | 0,25 | Default do modelo; neutro entre formulações |
| Variante SAM 2.1 | Hiera Large | Maior capacidade; isola efeito do prompt |
| GPU | NVIDIA A100 (40 GB) | Ambiente Azure Container Apps |
| Framework | PyTorch 2.x | Inferência sem fine-tuning |
| Matching | Algoritmo Húngaro (`scipy.optimize.linear_sum_assignment`) | Atribuição bipartida ótima |

### 7.8 Métricas de Avaliação (3 Camadas)

**Camada 1 — Localização (Grounding)**

| Métrica | Definição |
|---------|-----------|
| Box IoU | IoU entre box predita e box derivada da máscara GT |
| Taxa de Detecção | Proporção de instâncias GT com pelo menos uma box com IoU ≥ 0,50 |

**Camada 2 — Segmentação (Máscara)**

| Métrica | Definição |
|---------|-----------|
| Mask IoU (mIoU) | $\frac{\sum (M_{\text{pred}} \land M_{\text{gt}})}{\sum (M_{\text{pred}} \lor M_{\text{gt}})}$ |
| Dice Score | $\frac{2 |M_{\text{pred}} \cap M_{\text{gt}}|}{|M_{\text{pred}}| + |M_{\text{gt}}|}$ |
| Boundary F1 | F1-score sobre pixels de borda (tolerância de 2 pixels) |

**Camada 3 — Utilidade Prática**

| Categoria | Critério | Interpretação |
|-----------|----------|---------------|
| **Boa** | IoU ≥ 0,75 | Anotador apenas valida; economia de ~90% do tempo |
| **Corrigível** | 0,50 ≤ IoU < 0,75 | Anotador corrige; economia de ~50–70% |
| **Ruim** | IoU < 0,50 | Anotador refaz do zero; sem economia |

> **Nota**: Esses limiares constituem uma *heurística operacional* (proxy) para utilidade prática, não uma métrica validada por estudo de tempo com anotadores humanos. Os percentuais de economia são estimativas da literatura de anotação assistida (Mikulová et al., 2022), não medições diretas deste estudo.

### 7.9 Desenho Experimental

| Exp. | Objetivo | Variável Independente | Variável Dependente |
|------|----------|----------------------|---------------------|
| E1 | Qualidade-base por categoria | — (baseline) | mIoU, Dice, taxa de detecção, utilidade |
| E3 | Impacto da formulação do prompt | Tipo de prompt (4 níveis) | mIoU, Dice por tipo |
| E4 | Diagnóstico de bottleneck | Estágio (grounding vs. segmentação) | Box IoU vs. Mask IoU condicional |
| E5 | Taxonomia de falhas | — | Categorização de erros |

### 7.10 Unidade de Análise

A unidade primária de análise é o **par (instância GT, formulação de prompt)** — cada instância anotada no ground truth é avaliada independentemente para cada formulação de prompt, resultando em 18.360 observações (4.590 instâncias GT × 4 formulações). Para testes pareados, a unidade de pareamento é a **imagem** (500 imagens), com mIoU médio por imagem como variável resumo.

### 7.11 Análise Estatística

Para conferir rigor inferencial aos resultados, as seguintes técnicas foram empregadas:

1. **Intervalos de confiança por bootstrap** (10.000 reamostragens, seed=42): IC 95% para mIoU e taxa de detecção de cada formulação, calculados sobre as 4.590 instâncias GT por condição.
2. **Teste de Wilcoxon signed-rank** (pareado por imagem, alternativa unilateral): compara o mIoU médio por imagem do prompt simples contra cada formulação alternativa. A hipótese nula é $H_0: \text{mIoU}_{\text{simples}} \leq \text{mIoU}_{\text{alternativo}}$.
3. **Tamanho de efeito de Cliff's delta** ($\delta$): medida não-paramétrica do efeito da formulação, classificada como negligível ($|\delta| < 0,147$), pequeno ($< 0,33$), médio ($< 0,474$) ou grande ($\geq 0,474$).

Essa combinação — IC bootstrap para estimação, Wilcoxon para teste de hipótese, Cliff's delta para magnitude — evita premissas de normalidade e é robusta para distribuições assimétricas de IoU.

### 7.12 Controle de Reprodutibilidade

- **Semente fixa** (seed=42) para amostragem estratificada e bootstrap
- **Código open-source**: pipeline modular em 10 módulos Python
- **Dados brutos versionados**: 500 arquivos JSON com resultados por instância
- **Run de referência**: `run-20260515T011220Z`, GPU A100 (40 GB), Azure Container Apps

---

## 8. Resultados Obtidos

Os experimentos foram executados em 14–15 de maio de 2026, em ambiente Azure Container Apps com GPU NVIDIA A100 (40 GB). O run de referência (`run-20260515T011220Z`) processou 500 imagens × 10 classes × 4 formulações = **18.360 avaliações de instância**, totalizando 44.201 resultados individuais (incluindo falsos positivos e *misses*).

### 8.1 Qualidade-Base do Pipeline (E1)

O pipeline com prompts simples em inglês (*baseline*) alcança **mIoU médio de 0,527** sobre as 10 classes, com variação substancial entre categorias:

| Classe | mIoU | Dice | Boundary F1 | Taxa de Detecção | N instâncias |
|--------|------|------|-------------|------------------|-------------|
| cat | 0,826 | — | — | 92,6% | 54 |
| dog | 0,767 | — | — | 89,6% | 67 |
| person | 0,667 | — | — | 82,9% | 1.863 |
| bottle | 0,558 | — | — | 68,0% | 535 |
| pizza | 0,490 | — | — | 60,0% | 150 |
| car | 0,476 | — | — | 57,4% | 411 |
| cup | 0,372 | — | — | 43,0% | 512 |
| chair | 0,337 | — | — | 50,0% | 722 |
| bicycle | 0,303 | — | — | 48,0% | 171 |
| apple | 0,260 | — | — | 35,2% | 105 |

**Padrão observado**: Classes com alta saliência visual (cat, dog) atingem mIoU > 0,75. Classes com objetos pequenos, baixo contraste ou oclusão frequente (apple, bicycle, chair) apresentam mIoU < 0,35 e taxas de detecção abaixo de 50%.

**Distribuição de utilidade** (média das 10 classes com prompts simples):

| Classificação | Percentual |
|--------------|-----------|
| Boa (≥ 0,75) | ~36% |
| Corrigível (0,50–0,75) | ~19% |
| Ruim (< 0,50) | ~13% |
| Não detectada | ~33% |

### 8.2 Impacto da Formulação do Prompt (E3) — Resultado Central

**O resultado mais relevante é contraintuitivo**: o prompt simples — um único nome de classe — supera todas as formulações elaboradas por margem substancial.

| Formulação | mIoU | IC 95% | Taxa de Detecção | IC 95% Det. | Δ mIoU vs. Simples | p (Wilcoxon) | Cliff's δ |
|-----------|------|--------|------------------|-------------|---------------------|--------------|-----------|
| **Simples** | **0,527** | [0,515; 0,539] | **66,3%** | [64,9%; 67,7%] | — | — | — |
| Contextual | 0,250 | [0,239; 0,261] | 30,7% | [29,4%; 32,0%] | −0,277 (−52,6%) | 6,95 × 10⁻⁵⁵ | 0,43 (médio) |
| Direto | 0,210 | [0,200; 0,221] | 25,4% | [24,1%; 26,6%] | −0,317 (−60,2%) | 1,08 × 10⁻⁶⁶ | 0,52 (grande) |
| Orientado a obj. | 0,204 | [0,194; 0,214] | 24,8% | [23,6%; 26,1%] | −0,323 (−61,3%) | 5,40 × 10⁻⁶⁵ | 0,52 (grande) |

**Achados-chave**:
- Prompts simples produzem mIoU **2,5× superior** ao melhor alternativo
- A degradação se manifesta principalmente na **taxa de detecção** (de 66,3% para 24,8–30,7%)
- Quando o grounding é bem-sucedido (Box IoU ≥ 0,75), o SAM 2.1 entrega Mask IoU ≈ **0,84–0,86 independente da formulação**

**Interação Formulação × Classe** (exemplos extremos):

| Classe | Simples | Contextual | Direto | Objeto |
|--------|---------|-----------|--------|--------|
| cat | 0,826 | 0,808 | 0,805 | 0,727 |
| dog | 0,767 | 0,749 | 0,724 | 0,714 |
| person | 0,667 | 0,207 | 0,186 | 0,198 |
| bottle | 0,558 | 0,279 | 0,189 | 0,152 |
| chair | 0,337 | 0,200 | 0,183 | 0,187 |

Classes de alta saliência (cat, dog) são menos afetadas; classes de baixa saliência sofrem degradação amplificada — chegando a mIoU próximo de zero.

### 8.3 Diagnóstico de Bottleneck: Grounding vs. Segmentação (E4)

A análise condicional isola a contribuição de cada estágio:

| Faixa de Box IoU | Mask IoU médio | Dice médio | Boundary F1 médio |
|-----------------|----------------|-----------|-------------------|
| ≥ 0,75 (boa) | **0,849** | **0,914** | **0,728** |
| ≥ 0,50 (aceitável) | **0,828** | **0,899** | **0,717** |

**Resultado central**: Quando o bounding box é preciso (IoU ≥ 0,75), o SAM 2.1 produz Mask IoU = **0,849** — qualidade suficiente para pré-rotulagem automática. O bottleneck do pipeline está integralmente no grounding.

**Distribuição de erros por estágio** (N = 9.304 pares matched):

| Tipo de Erro | Percentual |
|-------------|-----------|
| **Grounding miss** (sem detecção) | **73,8%** |
| Falso positivo | 16,2% |
| Máscara incompleta (box OK, máscara ruim) | 7,4% |
| Box incompleto | 1,8% |
| Erro de segmentação puro (box bom, máscara ruim) | **2,2%** |

**75% das falhas são grounding misses** — o pipeline falha em detectar o objeto, não em segmentá-lo.

### 8.4 Taxonomia de Falhas (E5)

| Código | Tipo de Falha | Frequência |
|--------|--------------|-----------|
| F4 | Objeto pequeno (< 1% da área da imagem) | 24,1% |
| F1 | Oclusão total | 18,3% |
| F6 | Aparência atípica (ângulo/iluminação) | 17,1% |
| F3 | Baixo contraste com o fundo | 15,7% |
| F5 | Aglomeração (crowd) de mesma classe | 13,6% |
| F2 | Truncamento (objeto fora do frame) | 11,2% |

Classes com maior taxa de falha (chair, bottle) concentram F4 e F3 (objetos pequenos e baixo contraste). Classes com menor falha (cat, dog) são dominadas por F6 (ângulo atípico) e F5 (aglomeração) — padrões tratáveis com ajuste de threshold.

---

## 9. Discussão dos Resultados

### 9.1 Explicações Plausíveis para a Superioridade de Prompts Simples

Três mecanismos arquiteturais são propostos como *explicações plausíveis* para o resultado contraintuitivo. Estas hipóteses explicativas são consistentes com os dados observados e com a arquitetura documentada dos modelos, mas este estudo não analisa diretamente os mapas de atenção nem realiza ablação interna dos encoders — portanto, tratam-se de inferências arquiteturais, não de demonstrações causais.

**Explicação 1 — Diluição de atenção no BERT-base**: O Grounding DINO utiliza BERT-base como text encoder. No prompt `the dog in the image` (6 tokens), o token semanticamente informativo (`dog`) compete com tokens funcionais (`the`, `in`, `image`) pelo peso de atenção cruzada com as features visuais. No prompt simples `dog`, o token relevante concentra virtualmente toda a atenção cross-modal.

**Explicação 2 — Distribuição de treinamento**: O pré-treinamento do Grounding DINO usa a convenção de concatenar nomes de categorias simples separados por ponto (`cat . dog . person .`). Formulações como `segment the dog` ou `object: dog` estão fora da distribuição de treinamento.

**Explicação 3 — Separação de papéis**: O Grounding DINO é um localizador, não um seguidor de instruções. Formulações imperativas pressupõem capacidade de decomposição instrucional que não faz parte do design do modelo.

### 9.2 Verificação das Hipóteses

| Hipótese | Evidência Empírica | Veredicto |
|----------|-------------------|-----------|
| **H1** | Simple mIoU 0,527 [IC 95%: 0,515–0,539] vs. Object 0,204 [0,194–0,214]; Wilcoxon p = 5,40 × 10⁻⁶⁵; Cliff's δ = 0,52 (grande) | ✅ Confirmada |
| **H2** | Mask IoU condicional (Box ≥ 0,75): 0,838–0,861 independente de formulação; ICs sobrepostos entre prompts | ✅ Confirmada |
| **H3** | 73,8% das falhas são grounding misses; erro de segmentação puro = 2,2% | ✅ Confirmada |

### 9.3 Implicações Práticas

1. **Use prompts simples**: A evidência contradiz a intuição de que instruções mais detalhadas melhoram o desempenho.
2. **Invista no grounding**: Melhorias devem focar o estágio de localização textual, não a segmentação.
3. **Triagem por categoria**: Classes com taxa de detecção < 50% (chair, bottle, apple) têm ROI negativo para pré-rotulagem zero-shot.
4. **Framework de avaliação reutilizável**: O protocolo Detection → Mask quality → Practical utility aplica-se a qualquer pipeline VLM de pré-rotulagem.

---

## 10. Contribuições

1. **Evidência empírica contraintuitiva sobre formulação de prompts**: Prompts simples superam consistentemente formulações complexas em pipelines de grounding+segmentação, com diferença de 2,5× em mIoU — resultado diretamente acionável por praticantes.

2. **Diagnóstico de origem de erros via análise condicional**: Isolamento do impacto da etapa de grounding (text-to-box) da etapa de segmentação (box-to-mask), identificando que a degradação de qualidade é quase inteiramente determinada pela etapa de localização textual.

3. **Framework de avaliação em 3 camadas para pipelines de pré-rotulagem**: Métricas de localização, segmentação e utilidade prática como protocolo reutilizável, agnóstico aos modelos subjacentes.

---

## 11. Limitações e Trabalhos Futuros

### 11.1 Limitações

- **Generalização de modelos**: Resultados específicos para Grounding DINO (Swin-B) + SAM 2.1 (Hiera Large). O efeito *less is more* está mecanicamente vinculado ao text encoder BERT-base. Outros modelos de grounding com encoders distintos (CLIP, T5) podem apresentar sensibilidade diferente à formulação.
- **Ausência de validação por anotadores humanos**: As categorias de utilidade prática (Boa/Corrigível/Ruim) são definidas por limiares de IoU, não por estudo de tempo real com anotadores. A correlação entre IoU e economia efetiva de tempo não foi validada empiricamente neste trabalho.
- **Threshold fixo**: Os parâmetros `box_threshold=0.30` e `text_threshold=0.25` foram mantidos constantes. Thresholds otimizados por formulação ou por classe poderiam alterar os valores quantitativos, embora a direção do efeito (simples > complexo) seja improvável de inverter.
- **Dependência do inglês**: Todos os prompts foram formulados em inglês. O comportamento com prompts em outros idiomas não foi avaliado neste estudo.
- **Cobertura de classes**: 10 classes do COCO cobrem objetos comuns mas não domínios especializados (médico, industrial, aéreo).
- **Formulações testadas**: As 4 formulações representam o espaço mais comum, mas não esgotam as possibilidades (atributos visuais, âncoras espaciais).
- **Dataset único**: Todos os experimentos em COCO val2017. Validação em datasets complementares (LVIS, ADE20K) fortaleceria a generalidade dos resultados.

### 11.2 Trabalhos Futuros

1. **Modelos de grounding de segunda geração**: Grounding DINO 1.5 e Florence-2 utilizam text encoders mais robustos. A sensibilidade desses modelos à formulação é uma pergunta aberta.
2. **Ajuste fino do estágio de grounding**: Classes com taxa de detecção < 50% poderiam beneficiar-se de fine-tuning supervisionado com amostras existentes.
3. **Validação em domínios especializados**: Aplicação a domínios com distribuições visuais distintas do COCO.

---

## 12. Cronograma Realizado

| Fase | Período | Status |
|------|---------|--------|
| Definição do problema e revisão da literatura | Abr 2026 | ✅ Concluído |
| Projeto do pipeline e implementação | Mai 2026 (1ª semana) | ✅ Concluído |
| Deploy em Azure Container Apps (GPU A100) | Mai 2026 (2ª semana) | ✅ Concluído |
| Execução dos experimentos (run-20260515T011220Z) | 14–15 Mai 2026 | ✅ Concluído |
| Análise de resultados e pivot de framing | 15–21 Mai 2026 | ✅ Concluído |
| Redação das seções do artigo (Seções 1–6) | 21–29 Mai 2026 | ✅ Concluído |
| Consolidação LaTeX e revisão final | Jun 2026 | 🔲 Pendente |
| Submissão | Jun–Jul 2026 | 🔲 Pendente |

---

## 13. Infraestrutura e Reprodutibilidade

| Recurso | Especificação |
|---------|---------------|
| **GPU** | NVIDIA A100 (40 GB) — Azure Container Apps |
| **Modelos** | Grounding DINO Swin-B (`groundingdino_swinb_cogcoor.pth`, 938 MB) + SAM 2.1 Hiera Large (`sam2.1_hiera_large.pt`, 898 MB) |
| **Framework** | PyTorch 2.x, Python 3.10+ |
| **Dataset** | COCO val2017 (5.000 imagens, amostra de 500) |
| **Código** | Pipeline modular open-source (10 módulos Python) |
| **Matching** | Algoritmo Húngaro via `scipy.optimize.linear_sum_assignment` |

---

## Referências

- Cao, J. et al. (2024). MAPLM: A Real-World Large-Scale Vision-Language Benchmark for Map and Traffic Scene Understanding. *CVPR 2024*.
- COCONut (2024). Modernizing COCO Panoptic Segmentation Annotations.
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
- Ganguly, S. et al. (2025). Labeling Copilot: Agentic Data Curation for Computer Vision. *arXiv*.
- Infomineo (2024). Data Annotation: The Hidden Foundation of Enterprise AI.
- Kirillov, A. et al. (2023). Segment Anything. *ICCV 2023*.
- Kuhn, H. W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics Quarterly*.
- Liang, F. et al. (2023). Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP. *CVPR 2023*.
- Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*.
- Liu, S. et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. *ECCV 2024*.
- Lu, Y. et al. (2022). Prompt Distribution Learning. *CVPR 2022*.
- Mikulová, M. et al. (2022). Quality and Efficiency of Manual Annotation: Pre-annotation Bias. *LREC 2022*.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
- Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos. *arXiv*.
- Sultan, A. et al. (2025). GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Geo-Spatial Data. *ECAI 2025*.
- Yang, J. et al. (2020). Inter-annotator Agreement in Medical Image Segmentation. *Medical Image Analysis*.
- Zhang, J. et al. (2023). A Simple Framework for Open-Vocabulary Segmentation and Detection. *ICCV 2023*.
- Zhou, K. et al. (2022). Learning to Prompt for Vision-Language Models (CoOp). *IJCV 2022*.
- Zhou, X. et al. (2025). Open-Vocabulary Object Detection in UAV Imagery: A Survey. *Drones*.
- Zhu, D. et al. (2023). Aligning Perception with Language: Anchored Prompt Engineering. *NeurIPS 2023*.
