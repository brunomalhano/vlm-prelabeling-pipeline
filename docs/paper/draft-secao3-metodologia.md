# Seção 3 — Metodologia

> **Extensão-alvo**: ~1.400 palavras

---

## 3. Metodologia

A presente pesquisa é de natureza aplicada e experimental, com abordagem predominantemente quantitativa e componente qualitativo complementar. O objetivo é avaliar a viabilidade de um pipeline modular de pré-rotulagem de máscaras de segmentação baseado em modelos visão-linguagem, investigando o impacto da **formulação dos prompts textuais** na qualidade das máscaras geradas automaticamente.

### 3.1 Dataset

Os experimentos foram conduzidos sobre o conjunto de validação do COCO 2017 (*val2017*), composto por 5.000 imagens com anotações oficiais de segmentação por instância [1]. O COCO foi selecionado por três razões: (i) disponibilidade de máscaras de segmentação por instância com identificação de classe, permitindo comparação direta com as máscaras geradas pelo pipeline; (ii) ampla adoção como benchmark na literatura de segmentação e detecção, facilitando a comparação com trabalhos relacionados; e (iii) diversidade de classes e cenários visuais, incluindo objetos de diferentes escalas, contextos e graus de oclusão.

Para viabilizar a execução experimental em escopo controlado, foi selecionado um subconjunto de 500 imagens do *val2017* por amostragem estratificada por classe. O processo garantiu um mínimo de 30 instâncias por classe-alvo, assegurando representatividade estatística para comparações entre formulações de prompt. Imagens foram selecionadas priorizando aquelas que contivessem ao menos uma instância das classes definidas na Seção 3.2.

Como o COCO apresenta desbalanceamento natural de frequência entre classes (e.g., *person* muito mais frequente que *cat*/*dog*), a avaliação foi planejada com três lentes complementares de agregação (micro, macro e balanceada por classe+instância), descritas formalmente na Seção 3.9. Essa decisão metodológica evita que conclusões globais sejam artefatos da distribuição de classes.

### 3.2 Classes Selecionadas

Foram selecionadas 10 classes do COCO, organizadas em cinco grupos visuais que representam diferentes graus de complexidade para segmentação:

| Grupo Visual | Classe | Justificativa |
|-------------|--------|---------------|
| Pessoas | person | Classe mais frequente no COCO; alta variabilidade de pose e escala |
| Animais | dog | Formas orgânicas; bordas irregulares |
| Animais | cat | Similar a *dog*; permite comparação intragrupo |
| Veículos | car | Formas rígidas; bordas bem definidas |
| Veículos | bicycle | Estrutura vazada; desafiante para segmentação |
| Domésticos | chair | Variedade de formas; frequente oclusão parcial |
| Domésticos | bottle | Objeto pequeno; frequentemente agrupado |
| Domésticos | cup | Similar a *bottle*; permite controle intragrupo |
| Alimentos | apple | Objeto pequeno; semântica simples |
| Alimentos | pizza | Forma plana; bordas ambíguas contra o fundo |

A seleção abrange classes com diferentes características visuais (objetos rígidos e deformáveis, grandes e pequenos, com bordas nítidas e ambíguas), permitindo análise diferenciada da qualidade de pré-rotulagem por categoria.

### 3.3 Pipeline Experimental

O pipeline de pré-rotulagem é composto por duas etapas sequenciais, cada uma executada por um modelo fundacional distinto:

**Etapa 1 — Localização textual (Grounding DINO, Swin-B).** Dado um prompt textual $p$ e uma imagem $I$, o modelo Grounding DINO com backbone Swin-B [2] retorna um conjunto de bounding boxes candidatas $B = \{b_1, b_2, ..., b_n\}$, cada uma associada a um score de confiança $s_i \in [0, 1]$. O modelo utiliza um encoder Transformer com pré-treinamento grounded que alinha representações textuais e visuais para localizar objetos arbitrários a partir de descrições em linguagem natural. Foi utilizada a variante com backbone Swin-B, que é a versão de maior capacidade totalmente open-source, garantindo replicabilidade integral dos experimentos.

**Etapa 2 — Geração de máscara (SAM 2.1, Hiera Large).** Cada bounding box $b_i$ aceita na Etapa 1 é fornecida como prompt visual ao Segment Anything Model 2.1 [4], na variante Hiera Large. O SAM 2.1 gera uma máscara de segmentação $m_i$ correspondente à região delimitada pela box. A separação em duas etapas permite diagnosticar a origem de erros: se a box estiver incorreta, o problema reside no grounding textual; se a box estiver correta mas a máscara for inadequada, o problema está na segmentação.

O fluxo completo é:

$$\text{Imagem} + \text{Prompt} \xrightarrow{\text{GDINO Swin-B}} \text{Bounding Boxes} \xrightarrow{\text{SAM 2.1}} \text{Máscaras Candidatas} \xrightarrow{\text{Comparação}} \text{Métricas}$$

### 3.4 Formulação dos Prompts

Para cada classe-alvo, foram definidos quatro tipos de prompt em inglês, totalizando quatro condições experimentais por classe e 40 prompts distintos no total. O inglês é o único idioma analisado na comparação principal, pois o Grounding DINO utiliza BERT-base como text encoder — treinado predominantemente em inglês — e avaliar outros idiomas testaria uma limitação arquitetural conhecida, não a questão de formulação:

| Tipo de Prompt | Exemplo | Rationale |
|---------------|---------|-----------|
| **Simples** | `dog` | Formato padrão na literatura de open-vocabulary detection |
| **Direto** | `segment the dog` | Verbo de instrução explícito — mais comum em pipelines com LLM |
| **Contextual** | `the dog in the image` | Contexto visual explícito — alinha com templates de referência |
| **Orientado a objeto** | `object: dog` | Prefixo de tipo — testa sensibilidade a formato estruturado |

As formulações foram construídas manualmente pelo autor, mantendo equivalência semântica entre os tipos. O tipo *Simples* corresponde ao formato predominante na literatura e serve como baseline. Os tipos *Direto*, *Contextual* e *Orientado a objeto* testam se informações adicionais no prompt melhoram ou degradam a localização — e em qual etapa do pipeline o efeito se manifesta.

### 3.5 Configuração Experimental

Os seguintes parâmetros foram fixados para todos os experimentos:

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `box_threshold` | 0.30 | Equilíbrio entre precision e recall; ligeiramente acima do default (0.25) para reduzir falsos positivos |
| `text_threshold` | 0.25 | Default do modelo; valor neutro que não favorece nenhum tipo de formulação |
| Boxes por prompt | Todas acima do threshold | Cenário realista de pré-rotulagem: o anotador receberia todas as candidatas |
| Variante SAM 2.1 | Hiera Large | Variante de maior capacidade; maximiza qualidade da máscara para isolar o efeito da formulação do prompt |

**Ambiente computacional.** Todos os experimentos foram executados em ambiente de nuvem com GPU NVIDIA A100 (40 GB), utilizando PyTorch 2.x como framework de inferência. Os modelos foram carregados com pesos pré-treinados oficiais sem fine-tuning adicional. O código-fonte do pipeline, incluindo scripts de avaliação e geração de prompts, será disponibilizado publicamente para garantir replicabilidade.

**Matching de instâncias.** Em cada imagem, o pipeline pode gerar $M$ máscaras preditas para uma classe que possui $N$ instâncias no ground truth do COCO. A avaliação requer associar cada predição a exatamente uma instância GT (ou classificá-la como falso positivo), e cada GT a no máximo uma predição (ou classificá-la como *miss*). Esse problema constitui uma **atribuição bipartida ótima** — dado um grafo bipartido cujos nós são predições e instâncias GT, com pesos nas arestas dados pelo IoU entre as máscaras, busca-se o emparelhamento de peso máximo que respeite a restrição one-to-one.

Uma abordagem greedy — que associa iterativamente o par de maior IoU — pode produzir atribuições globalmente subótimas. Considere o caso em que uma predição $p_1$ tem IoU 0,82 com $g_1$ e 0,78 com $g_2$, enquanto $p_2$ tem IoU 0,80 com $g_1$ e 0,40 com $g_2$. O matching greedy associaria $p_1 \to g_1$ (0,82) e $p_2 \to g_2$ (0,40), totalizando 1,22. O matching ótimo associaria $p_1 \to g_2$ (0,78) e $p_2 \to g_1$ (0,80), totalizando 1,58. Em cenários com múltiplas instâncias sobrepostas — frequentes no COCO — essa diferença impacta diretamente as métricas de segmentação.

O **algoritmo Húngaro** [15, 16] resolve o problema de atribuição bipartida em tempo polinomial $O(n^3)$, garantindo a solução globalmente ótima. O algoritmo é o padrão de facto para avaliação de detecção e segmentação de instâncias, sendo utilizado tanto no protocolo de avaliação oficial do COCO [1] quanto em métodos baseados em Transformer como o DETR [17], que adota o matching bipartido Húngaro como função de perda durante o treinamento.

Neste trabalho, a implementação utiliza `scipy.optimize.linear_sum_assignment`, que resolve o problema na sua formulação de minimização. A matriz de custo $\mathbf{C} \in \mathbb{R}^{M \times N}$ é construída como:

$$C_{ij} = 1 - \text{IoU}(\hat{m}_i, m_j^{GT})$$

onde $\hat{m}_i$ é a $i$-ésima máscara predita e $m_j^{GT}$ é a $j$-ésima máscara ground truth da mesma classe na mesma imagem. A atribuição ótima $\sigma^* = \arg\min_\sigma \sum_i C_{i,\sigma(i)}$ é então filtrada: pares com $\text{IoU} < 0{,}10$ são descartados (a predição é reclassificada como falso positivo). Predições sem correspondência são falsos positivos; instâncias GT sem correspondência são *misses* (instâncias não detectadas). A escolha do limiar mínimo de 0,10 evita associações espúrias entre predições e GTs que compartilham apenas sobreposição marginal, sem penalizar excessivamente predições parcialmente corretas que ainda teriam utilidade em um cenário de pré-rotulagem assistida.

### 3.6 Métricas de Avaliação

A avaliação é organizada em três camadas, cada uma capturando um aspecto distinto da qualidade do pipeline:

**Camada 1 — Localização (Grounding).** Avalia se o Grounding DINO localizou corretamente o objeto solicitado.

| Métrica | Definição |
|---------|-----------|
| Box IoU | Intersection over Union entre a bounding box gerada e a bounding box derivada da máscara GT |
| Grounding Success Rate (GSR) | Proporção de instâncias GT que receberam ao menos uma box com IoU ≥ 0.50 |
| Confidence Score | Score médio de confiança retornado pelo GDINO para boxes aceitas |

**Camada 2 — Segmentação (Máscara).** Avalia a qualidade da máscara gerada pelo SAM em relação à máscara oficial do COCO.

| Métrica | Definição |
|---------|-----------|
| Mask IoU (mIoU) | IoU médio entre máscaras geradas e GT, calculado por classe e agregado |
| Dice Score | $\frac{2 |P \cap G|}{|P| + |G|}$, onde $P$ é a máscara predita e $G$ é o ground truth |
| Boundary F1 | F1-score calculado sobre os pixels de borda da máscara (tolerância de 2 pixels) |

**Camada 3 — Utilidade Prática.** Classifica cada máscara gerada em uma categoria de utilidade para anotação assistida, baseada no Mask IoU:

| Categoria | Critério | Interpretação |
|-----------|----------|---------------|
| **Boa** | IoU ≥ 0.75 | O anotador provavelmente apenas valida a máscara |
| **Corrigível** | 0.50 ≤ IoU < 0.75 | O anotador corrige a máscara, mas economiza esforço em relação à criação manual |
| **Ruim** | IoU < 0.50 | O anotador provavelmente refaz a máscara do zero |

As faixas foram definidas a priori com base em critérios operacionais de anotação assistida. O objetivo não é apenas obter uma métrica média, mas estimar a distribuição de utilidade prática das máscaras como pré-rótulos, permitindo avaliar o potencial de redução de esforço humano em cenários reais.

### 3.7 Desenho Experimental

Os cinco experimentos seguem uma progressão do geral para o específico:

| Exp. | Objetivo | Variável Independente | Variável Dependente |
|------|----------|----------------------|---------------------|
| E1 | Qualidade geral do pipeline | — (baseline) | mIoU, Dice, GSR, distribuição boa/corrigível/ruim |
| E3 | Impacto da formulação do prompt | Tipo de prompt (4 níveis) | mIoU, Dice por tipo de prompt |
| E4 | Diagnóstico de fonte de erro | Etapa do pipeline (grounding vs segmentação) | Box IoU vs Mask IoU (análise condicional) |
| E5 | Taxonomia de falhas | — | Categorização de erros |

> Nota de versionamento do desenho: a comparação multilíngue (E2) foi removida da análise principal após pivot metodológico para foco em engenharia de prompts no cenário EN-only. Análises de idioma permanecem como apêndice opcional quando executadas.

O Experimento 4 é central para a Hipótese H2. A análise condiciona o Mask IoU ao Box IoU em dois limiares alinhados ao protocolo de avaliação do COCO [1]: IoU ≥ 0.50 (detecção correta, critério padrão) e IoU ≥ 0.75 (detecção estrita). Para instâncias onde Box IoU ≥ 0.75, calcula-se o Mask IoU separadamente. Se este permanecer alto, a degradação de qualidade observada nos demais casos é atribuível ao grounding, não à segmentação. A análise no limiar de 0.50 permite avaliar a proporção de casos onde o grounding é ao menos razoável.

### 3.8 Unidade de Análise

A unidade primária de análise é o **par (instância GT, formulação de prompt)** — cada instância anotada no ground truth é avaliada independentemente para cada formulação, resultando em 18.360 observações (4.590 instâncias GT × 4 formulações). Para testes pareados, a unidade de pareamento é a **imagem** (500 imagens), com mIoU médio por imagem como variável resumo.

### 3.9 Análise Estatística

Para conferir rigor inferencial aos resultados, a análise foi estruturada em três níveis de agregação e três técnicas inferenciais complementares.

**Níveis de agregação reportados (robustez a desbalanceamento):**

1. **Micro-average (ponderado por instância):**
$$
	ext{mIoU}_{micro} = \frac{1}{N}\sum_{i=1}^{N} \text{IoU}_i
$$
onde $N$ é o total de instâncias GT avaliadas na condição.

2. **Macro-average (média por classe):**
$$
	ext{mIoU}_{macro} = \frac{1}{K}\sum_{k=1}^{K} \left(\frac{1}{N_k}\sum_{i \in k} \text{IoU}_i\right)
$$
onde $K$ é o número de classes e $N_k$ o número de instâncias da classe $k$.

3. **Avaliação balanceada por classe+instância:** para cada formulação, aplica-se subamostragem estratificada sem reposição para igualar o número de instâncias por classe ao menor $N_k$ observado naquela formulação. Esse protocolo controla simultaneamente a contribuição de classe e o tamanho amostral por classe.

**Técnicas inferenciais aplicadas:**

1. **Intervalos de confiança por bootstrap** (10.000 reamostragens, seed=42): IC 95% para mIoU e taxa de detecção de cada formulação, reportados para micro-average, macro-average e cenário balanceado por classe+instância.
2. **Teste de Wilcoxon signed-rank** (pareado por imagem, alternativa unilateral): compara o mIoU médio por imagem do prompt simples contra cada formulação alternativa. A hipótese nula é $H_0: \text{mIoU}_{\text{simples}} \leq \text{mIoU}_{\text{alternativo}}$.
3. **Tamanho de efeito de Cliff's delta** ($\delta$): medida não-paramétrica classificada como negligenciável ($|\delta| < 0{,}147$), pequeno ($< 0{,}33$), médio ($< 0{,}474$) ou grande ($\geq 0{,}474$).

Essa combinação evita premissas de normalidade, é robusta para distribuições assimétricas de IoU e reduz risco de interpretação enviesada por desbalanceamento de classes.

### 3.10 Controle de Reprodutibilidade

- **Semente fixa** (seed=42) para amostragem estratificada e bootstrap
- **Código open-source**: pipeline modular em 10 módulos Python
- **Dados brutos versionados**: 500 arquivos JSON com resultados por instância
- **Run de referência**: `run-20260515T011220Z`, GPU A100 (40 GB), Azure Container Apps

> **Nota sobre categorias de utilidade**: Os limiares de IoU para classificação em Boa/Corrigível/Ruim constituem uma *heurística operacional* (proxy), não uma métrica validada por estudo de tempo com anotadores humanos. Os percentuais de economia referidos na literatura de anotação assistida [18] não foram medições diretas deste estudo.
