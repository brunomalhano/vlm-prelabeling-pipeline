# Coaching Artifact — Article Outline & Concept Matrix

**Tema**: Pré-Rotulagem de Máscaras de Segmentação com Modelos Visão-Linguagem: Impacto da Formulação de Prompts no COCO Dataset  
**Título provisório**: *Less is More: Impact of Prompt Formulation on Pre-Labeling Quality with Grounding DINO and SAM 2.1*  
**Autor**: Bruno Malhano  
**Destino**: Artigo de disciplina (Tópicos Especiais em Deep Learning) + submissão para conferência/journal  
**Data**: 2026-05-12 (pivot: 2026-05-21)  
**Coach**: ArticleCoach  
**Fase atual**: Pivot de framing confirmado — idioma PT descartado como variável primária; formulação de prompt promovida a variável principal

---

## 1. Hipóteses Formalizadas

> **Decisão de pivot (2026-05-21)**: A comparação EN vs PT foi descartada como variável primária. O Grounding DINO utiliza BERT-base como text encoder — treinado predominantemente em inglês — tornando a degradação com PT uma consequência arquitetural previsível, não uma hipótese testável. A variável primária passa a ser a **formulação do prompt** em inglês, o que tem valor prático direto e lacuna real na literatura.

As hipóteses a seguir são operacionalizáveis e testáveis com os dados já coletados (experimento E3, EN only):

> **H1**: Prompts simples (single-word, e.g. "dog") produzem mIoU significativamente superior a formulações complexas (e.g. "the dog in the image", "segment the dog", "object: dog") no pipeline Grounding DINO + SAM 2.1 sobre o COCO val2017.

> **H2**: O efeito da complexidade do prompt concentra-se na etapa de grounding (box IoU), e não na etapa de segmentação (mask IoU condicionado a box IoU correto ≥ 0.50).

> **H3**: A principal fonte de falhas no pipeline é o grounding miss (ausência de detecção), não erros de qualidade de máscara.

**Status de verificação empírica (dados de E3/E4/E5):**

| Hipótese | Resultado | Evidência |
|----------|-----------|----------|
| H1 | ✅ Confirmada | simple (EN): 0.445 vs. object (EN): 0.180 — delta 2.5× |
| H2 | ✅ Confirmada | mask IoU = 0.836 com box IoU ≥ 0.50; segmentation errors = 1.5% |
| H3 | ✅ Confirmada | grounding_miss = 75.1% das falhas; grounding errors = 48.83% dos pares |

### Variáveis experimentais

| Variável | Tipo | Valores |
|----------|------|---------|
| Formulação do prompt | **Independente (primária)** | Simples, direto, contextual, orientado a objeto |
| Classe COCO | Moderadora | 10 classes selecionadas |
| Box IoU | Dependente (etapa 1) | Contínua [0, 1] |
| Mask IoU / mIoU | Dependente (etapa 2) | Contínua [0, 1] |
| Dice Score | Dependente (etapa 2) | Contínua [0, 1] |
| Utilidade prática | Dependente (derivada) | Boa (≥0.75), Corrigível [0.50–0.75), Ruim (<0.50) |
| Idioma | Secundária / controle | EN (análise principal); PT descartado como primária |

---

## 2. Concept Matrix

Eixos conceituais:

| Código | Eixo | Descrição |
|--------|------|-----------|
| **A** | Segmentação open-vocabulary | Métodos que segmentam a partir de descrições textuais, incluindo classes não vistas no treino |
| **B** | Grounding textual (text-to-box) | Modelos que localizam objetos via prompts em linguagem natural |
| **C** | Segmentação promptável (SAM-like) | Modelos que geram máscaras a partir de prompts visuais (boxes, pontos) |
| **D** | Pré-rotulagem / anotação assistida | Workflows que reduzem esforço humano na criação de datasets rotulados |
| **E** | Engenharia de prompts para VLMs | Impacto da formulação textual (tipo, comprimento, estrutura) no desempenho de modelos visão-linguagem em tarefas downstream |

> **Nota pós-pivot**: O Eixo E foi reorientado de "multilíngue" para "engenharia de prompts". Referências multilíngues (AltCLIP, NLLB-CLIP, XM3600) são removidas da matriz principal. Adicionar referências sobre prompt sensitivity em grounding/detecção (e.g., Zhong et al. 2022 RegionCLIP; APE, PromptDet, ou estudos de sensibilidade em CLIP).

### Matriz

| # | Fonte | Ano | Venue | A | B | C | D | E | Método |
|---|-------|-----|-------|---|---|---|---|---|--------|
| 1 | Liang et al. — Mask-adapted CLIP | 2023 | CVPR | ✓ | | | | | Two-stage OV seg + CLIP adaptation |
| 2 | Liu et al. — Grounding DINO | 2023/2024 | ECCV | | ✓ | | | | Transformer open-set detector |
| 3 | Kirillov et al. — Segment Anything (SAM) | 2023 | ICCV | | | ✓ | | | Foundation model, promptable seg |
| 4 | Lin et al. — Microsoft COCO | 2014 | ECCV | | | | | | Benchmark dataset |
| 5 | Zhang et al. — Simple Framework for OV Seg & Det | 2023 | ICCV | ✓ | ✓ | | | | Unified OV seg+det framework |
| 6 | Zhou et al. — OV Object Detection in UAV Imagery (Review) | 2025 | Drones | ✓ | ✓ | | | | Survey OVOD + aerial scenes |
| 7 | Snaebjarnarson et al. — Taxonomy-Aware Eval of VLMs | 2025 | CVPR | | | | | ✓ | Evaluation taxonomy for VLMs |
| 8 | Cao et al. — MAPLM | 2024 | CVPR | | ✓ | | | | VL benchmark for map/traffic |
| 9 | Sultan et al. — GeoSAM | 2023/2025 | ECAI | | ✓ | ✓ | | | Fine-tuning SAM with multi-modal prompts |
| 10 | SPIE — Is SAM 2 better than SAM in medical seg? | 2025 | SPIE | | | ✓ | | | SAM vs SAM 2 comparison |
| 11 | Ganguly et al. — Labeling Copilot | 2025 | arXiv | | ✓ | ✓ | ✓ | | Agentic data curation (GDINO+DETIC+SAM) |
| 12 | Mikulová et al. — Pre-annotation Bias | 2022 | LREC | | | | ✓ | | Pre-annotation quality/efficiency |
| 13 | Infomineo — Data Annotation Enterprise AI | 2024 | Report | | | | ✓ | | Annotation as strategic foundation |
| 14 | COCONut — Modernizing COCO Segmentation | 2024 | — | | | | ✓ | | Dataset quality, annotation consistency |
| 15 | Polygon-RNN++ | 2018 | CVPR | | | | ✓ | | Interactive annotation for masks |
| 16 | Springer — Inter-annotator agreement | 2011 | LRE | | | | ✓ | | Annotation consistency measurement |
| 17 | Nature — Medical imaging (s41746-026-02422-x) | 2026 | npj DM | | | ✓ | ✓ | | Medical seg + human-in-the-loop |
| 18 | IEEE 11286214 | 2025 | IEEE | ? | ? | ? | ? | ? | *(não acessível — verificar)* |
| 19 | IEEE 10883307 | 2025 | IEEE | ? | ? | ? | ? | ? | *(não acessível — verificar)* |
| 20 | ScienceDirect (PRL) | 2025 | PRL | ? | ? | ? | ? | ? | *(não acessível — verificar)* |
| **21** | **Chen et al. — AltCLIP** | **2022** | **arXiv** | | | | | **✓** | **Substitui text encoder do CLIP por XLM-R multilíngue; SOTA em ImageNet-CN, Flickr30k-CN, COCO-CN, XTD** |
| **22** | **Visheratin — NLLB-CLIP** | **2023** | **arXiv** | | | | | **✓** | **CLIP com text encoder NLLB (201 idiomas); supera SOTA em low-resource languages; benchmark XTD200 e Flickr30k-200** |
| **23** | **Thapliyal et al. — Crossmodal-3600 (XM3600)** | **2022** | **EMNLP** | | | | | **✓** | **Benchmark multilíngue: 3.600 imagens, 36 idiomas (inclui PT), captions humanas, avaliação de modelos multilíngues** |

### Diagnóstico da Matriz (Atualizado)

| Eixo | Papers | Cobertura | Status |
|------|--------|-----------|--------|
| A — Seg OV | 3 (#1, #5, #6) | ⚠️ Boa, mas considerar 1 método 2024–2025 | Opcional: adicionar SAN, FC-CLIP ou CAT-Seg |
| B — Grounding | 5 (#2, #5, #6, #8, #9) | ✅ Sólida | Suficiente |
| C — SAM-like | 5 (#3, #9, #10, #11, #17) | ✅ Sólida | Considerar SAM 2 (Ravi et al., 2024) |
| D — Pré-rotulagem | 7 (#11, #12, #13, #14, #15, #16, #17) | ✅ Forte | Labeling Copilot reforça muito |
| E — Multilíngue | **4 (#7, #21, #22, #23)** | **✅ Resolvido** | AltCLIP + NLLB-CLIP + XM3600 + Snaebjarnarson |

### Como usar os novos papers (Eixo E) no artigo

**AltCLIP (Chen et al., 2022)** — arXiv: 2211.06679
- *Onde citar*: Seção 2.5 (VLMs multilíngues), Seção 1 (Introdução — gap multilíngue)
- *Argumento*: Demonstra que substituir o text encoder do CLIP por um multilíngue (XLM-R) mantém desempenho próximo ao CLIP original em EN, com ganhos significativos em CN. **Mas**: não avalia impacto em tarefas downstream como grounding ou segmentação — apenas retrieval/classification.
- *Implicação para seu trabalho*: Se o text encoder multilíngue preserva qualidade em retrieval, será que preserva também em grounding textual (GDINO)? **Essa é exatamente a pergunta que seu trabalho responde empiricamente.**

**NLLB-CLIP (Visheratin, 2023)** — arXiv: 2309.01859
- *Onde citar*: Seção 2.5, Discussão
- *Argumento*: Modelos multilíngues como NLLB-CLIP cobrem 201 idiomas (inclui PT) e superam SOTA em low-resource languages. Porém, o Grounding DINO **não usa NLLB-CLIP** — usa seu próprio text encoder (BERT-based). Então a questão é: o embedding textual interno do GDINO lida bem com PT?
- *Implicação para seu trabalho*: Evidencia que o problema multilíngue está sendo resolvido no nível de CLIP/retrieval, mas **não foi levado para o nível de grounding + segmentação** — exatamente onde você contribui.

**Crossmodal-3600 / XM3600 (Thapliyal et al., 2022)** — EMNLP 2022
- *Onde citar*: Seção 2.5, Metodologia (justificativa de avaliar PT)
- *Argumento*: Benchmark com 36 idiomas (inclui português) para avaliação de modelos visão-linguagem. Mostra que desempenho varia significativamente entre idiomas. **Português está no benchmark**, o que valida a relevância da sua investigação.
- *Implicação para seu trabalho*: O XM3600 avalia captioning/retrieval, não grounding+segmentação. Seu trabalho estende a avaliação multilíngue para uma tarefa downstream mais complexa.

### Lacunas Residuais

1. **Papers inacessíveis** (#18, #19, #20) — Verificar e classificar manualmente; substituir se não forem essenciais
2. **Eixo A** — Opcional: adicionar 1 survey/método recente de OV segmentation (2024–2025)
3. **SAM 2** — Considerar adicionar Ravi et al. (2024) como referência complementar no Eixo C

---

## 3. Gap Statement (Rascunho)

> Trabalhos recentes demonstram que pipelines compostos por modelos de grounding textual (Grounding DINO) e segmentação promptável (SAM) podem gerar propostas de anotação automatizadas de qualidade competitiva em benchmarks como COCO (Ganguly et al., 2025). Paralelamente, a segmentação open-vocabulary avançou significativamente com métodos que alinham regiões visuais a descrições textuais (Liang et al., 2023; Zhang et al., 2023). No entanto, **nenhum estudo avaliou sistematicamente o impacto do idioma e da formulação dos prompts textuais na qualidade das máscaras geradas por esses pipelines modulares**. Em particular, o desempenho com prompts em português — relevante para equipes de anotação e domínios especializados em países lusófonos — permanece não investigado. Este trabalho preenche essa lacuna ao avaliar um pipeline Grounding DINO + SAM com prompts em português e inglês, medindo não apenas acurácia média, mas utilidade prática dos pré-rótulos para redução de esforço humano.

---

## 4. Outline Detalhado do Artigo

**Extensão-alvo**: ~7.000–8.000 palavras (compatível com journal; recortável para ~5.000 em conferência)

### Seção 1 — Introdução (~1.000 palavras, ~13%)

**Propósito**: Contextualizar, declarar o gap e a contribuição.

**Estrutura SCQA**:
- **S** (Situação): Visão computacional depende de dados rotulados; segmentação exige máscaras pixel-level que são caras e inconsistentes entre anotadores. Modelos fundacionais (CLIP, SAM, Grounding DINO) abrem possibilidade de pré-rotulagem automática.
- **C** (Complicação): Pipelines modulares (GDINO + SAM) mostram resultados promissores, mas todo o ecossistema de avaliação assume prompts em inglês. Equipes em contextos não-anglófonos usam taxonomias e guidelines em idioma local.
- **Q** (Questão): Modelos visão-linguagem podem gerar pré-rótulos úteis com prompts em português? Onde está o bottleneck — grounding ou segmentação?
- **A** (Resposta/Contribuição): Avaliação sistemática do pipeline GDINO + SAM com prompts em PT e EN, com métricas em 3 camadas (grounding, segmentação, utilidade prática).

**Argumento-chave**: O gap não é apenas técnico (acurácia de modelos), mas operacional (viabilidade de adoção em contextos multilíngues).

**Evidências necessárias**: Custo de anotação (Infomineo, E1–E3), foundation models (Kirillov, Liu), gap multilíngue (Snaebjarnarson).

**Contribuições declaradas** (lista numerada):
1. Pipeline de pré-rotulagem modular avaliado com métricas em 3 camadas
2. Primeira avaliação sistemática de prompts PT vs EN para grounding + segmentação
3. Classificação de utilidade prática (boa/corrigível/ruim) como proxy de redução de esforço

---

### Seção 2 — Trabalhos Relacionados (~1.500 palavras, ~20%)

**Propósito**: Posicionar o trabalho na interseção dos 5 eixos.

**Organização temática** (NÃO author-by-author):

| Subseção | Foco | Papers principais | Parágrafo de transição |
|----------|------|-------------------|----------------------|
| 2.1 Segmentação open-vocabulary | Evolução de closed-set para OV | Liang (2023), Zhang (2023), Zhou (2025 survey) | "Embora OV seg avance, a avaliação pressupõe prompts em inglês..." |
| 2.2 Grounding textual e detecção open-set | GDINO, GeoSAM, MAPLM | Liu (2023), Sultan (2025), Cao (2024) | "O grounding depende da qualidade do embedding textual, que pode variar com o idioma..." |
| 2.3 Segmentação promptável | SAM, SAM 2, extensões | Kirillov (2023), SPIE SAM2 (2025) | "SAM aceita prompts visuais, mas a qualidade da box de entrada é determinante..." |
| 2.4 Anotação assistida e pré-rotulagem | Pipelines, human-in-the-loop, curadoria | Ganguly (2025), Mikulová (2022), Polygon-RNN++, Infomineo | "Labeling Copilot demonstra que ensembles podem produzir pseudo-labels robustos, mas não avalia o impacto do idioma..." |
| 2.5 Modelos visão-linguagem multilíngues | CLIP multilíngue, avaliação cross-lingual | Snaebjarnarson (2025), **[buscar 2–3 papers]** | "A lacuna identificada é: nenhum estudo avaliou prompts PT em pipeline de pré-rotulagem..." |

**Cada subseção termina com uma observação crítica** conectando ao gap.

---

### Seção 3 — Metodologia (~1.400 palavras, ~18%) ✅ RASCUNHO COMPLETO

**Arquivo**: `platform/vlm-pipeline/docs/paper/draft-secao3-metodologia.md`

**Decisões confirmadas**:
- **Grounding DINO**: Original, Swin-B (open-source)
- **SAM**: 2.1, ViT-H (open-source)
- **Amostragem**: Estratificada (≥30 instâncias/classe em 500 imagens)
- **Thresholds**: box=0.30, text=0.25
- **Matching**: Hungarian (scipy.optimize.linear_sum_assignment)
- **Hardware**: GPU NVIDIA A100 (40GB), nuvem, PyTorch 2.x
- **E4 thresholds**: IoU ≥ 0.50 (padrão COCO) e ≥ 0.75 (estrito)

---

### Seção 4 — Experimentos e Resultados (~1.500 palavras, ~20%)

**Propósito**: Apresentar achados objetivamente, sem interpretação.

| Subseção | Hipótese testada | Saída esperada |
|----------|-----------------|----------------|
| 4.1 Qualidade geral do pipeline | Baseline | Tabela: mIoU, Dice, Box IoU médios por classe |
| 4.2 Inglês vs Português (H1) | H1 | Tabela comparativa EN vs PT por classe; teste estatístico |
| 4.3 Formulação do prompt (H2) | H2 | Tabela comparativa por tipo de prompt; gráfico de barras |
| 4.4 Grounding vs Segmentação (H3) | H3 | Scatter plot Box IoU × Mask IoU; análise condicional |
| 4.5 Utilidade prática | — | Distribuição boa/corrigível/ruim por idioma e classe |
| 4.6 Análise qualitativa de falhas | — | Exemplos visuais categorizados (5–8 figuras) |

**Regra**: Sem interpretação na seção de resultados. Tabelas e figuras devem ser auto-explicativas com caption.

---

### Seção 5 — Discussão (~1.200 palavras, ~16%)

**Propósito**: Interpretar, comparar com literatura, discutir limitações.

| Subseção | Foco |
|----------|------|
| 5.1 Interpretação das hipóteses | H1/H2/H3 confirmadas ou refutadas? Com que magnitude? |
| 5.2 Comparação com literatura | Como os resultados se situam vs. Labeling Copilot (Ganguly), Mask-adapted CLIP (Liang)? |
| 5.3 Implicações práticas | O que isso significa para equipes de anotação que operam em PT? |
| 5.4 Limitações | Tamanho do sample (500 imgs), COCO como proxy limitado, análise indireta de esforço |
| 5.5 Ameaças à validade | Validade interna (thresholds), externa (generalização para outros domínios), de constructo (IoU como proxy) |

---

### Seção 6 — Conclusão (~400 palavras, ~5%)

**Propósito**: Responder à pergunta de pesquisa, sintetizar contribuições, apontar futuro.

- Restatar contribuições (sem repetir abstract)
- Resposta direta à pergunta de pesquisa
- Trabalhos futuros específicos: (1) expandir para COCO val completo, (2) avaliar SAM 2, (3) adicionar domínios especializados (médico, industrial), (4) estudo cronometrado com anotadores reais

---

### Resumo / Abstract (~250 palavras, ~3%)

**Estrutura CPARI** — redigir por último:

| Elemento | Conteúdo |
|----------|----------|
| **C** (Contexto) | Segmentação exige máscaras pixel-level caras; modelos fundacionais possibilitam pré-rotulagem |
| **P** (Problema) | Nenhum estudo avaliou impacto de prompts em português nesse pipeline |
| **A** (Abordagem) | Pipeline GDINO + SAM avaliado com 8 condições de prompt (4 tipos × 2 idiomas) sobre COCO val |
| **R** (Resultados) | [após execução dos experimentos] |
| **I** (Implicações) | Direcionamento para equipes de anotação em contextos multilíngues |

---

## 5. Próximas Ações

| # | Ação | Prioridade | Fase |
|---|------|------------|------|
| 1 | **Buscar 2–3 papers sobre VLMs multilíngues** (CLIP multilíngue, avaliação cross-lingual) — gap crítico no Eixo E | 🔴 Alta | Fase 2 |
| 2 | **Verificar e classificar os 3 papers inacessíveis** (IEEE 11286214, IEEE 10883307, ScienceDirect PRL) | 🟡 Média | Fase 2 |
| 3 | **Considerar adicionar SAM 2 (Ravi et al., 2024)** como referência no Eixo C | 🟡 Média | Fase 2 |
| 4 | **Redigir Seção 2 (Trabalhos Relacionados)** seguindo a organização temática acima | 🔴 Alta | Fase 5 |
| 5 | **Implementar o pipeline** e executar os 5 experimentos para ter dados reais | 🔴 Alta | Execução |
| 6 | **Redigir Seções 1 e 3** (Introdução e Metodologia podem ser escritas antes dos resultados) | 🟡 Média | Fase 5 |
| 7 | **Definir venue-alvo** para calibrar formato (IEEE, SBC, ABNT) e extensão | 🟡 Média | Fase 6 |

---

## 6. Questão Socrática para Reflexão

> O **Labeling Copilot** (Ganguly et al., 2025) usa um ensemble de 3 modelos (DETIC, GroundingDINO, OWL-ViT) com mecanismo de consenso e atinge 37.1% mAP no COCO. Seu trabalho usa apenas GDINO + SAM (2 modelos, sem consenso). **Isso é uma limitação ou um design choice?** Se for design choice, como você argumenta que um pipeline mais simples pode ser mais relevante para o cenário de pré-rotulagem assistida?

*Dica*: A resposta está no posicionamento — Labeling Copilot busca anotação autônoma; seu trabalho busca pré-rotulagem para validação humana. São objetivos diferentes. Mas você precisa tornar essa distinção explícita no artigo.
