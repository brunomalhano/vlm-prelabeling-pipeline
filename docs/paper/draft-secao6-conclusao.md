# Seção 6 — Conclusão e Abstract

> **Extensão-alvo**: Conclusão ~600 palavras · Abstract ~250 palavras

---

## Abstract

> *Approximately 250 words.*

Pre-labeling with Vision-Language Models (VLMs) can reduce annotation cost in segmentation datasets, but prompt formulation is usually chosen without empirical evidence. We evaluate how prompt formulation affects instance-mask quality in a modular Grounding DINO (Swin-B) + SAM 2.1 (Hiera Large) pipeline on 500 COCO val2017 images across 10 classes.

We compare four formulations of the same class names: simple, direct, contextual, and object-prefixed. Results are counter-intuitive: simple prompts perform best in all aggregation views, reaching mIoU 0.527 [95% CI: 0.515–0.539] in micro-average, 0.506 [0.489–0.522] in macro-average, and 0.513 [0.477–0.547] under strict class+instance balancing. Alternative formulations are consistently inferior. Differences are statistically significant under paired Wilcoxon signed-rank tests ($p < 10^{-54}$), with medium-to-large effect sizes (Cliff's $\delta$ = 0.43–0.52).

Stage-wise decomposition shows that 73.8% of failures are grounding misses, while pure segmentation failures account for only ~2.2%. When grounding succeeds (box IoU ≥ 0.75), SAM 2.1 attains Mask IoU 0.849, indicating that segmentation quality is sufficient for practical pre-labeling once localization is correct.

These findings provide actionable guidance: use single class-name prompts as default, prioritize optimization of the grounding stage, and perform class-level ROI triage before large-scale pre-labeling deployment.

**Keywords**: instance segmentation, pre-labeling, Vision-Language Models, Grounding DINO, SAM 2.1, prompt formulation, zero-shot detection.

---

## 6 Conclusão

Este trabalho investigou o impacto da formulação de prompts textuais na qualidade de pré-rotulagem automática de máscaras de segmentação por um pipeline modular baseado em VLMs. A pergunta de pesquisa — qual formulação de prompt produz melhores pré-rótulos em um pipeline Grounding DINO + SAM 2.1? — recebeu uma resposta empiricamente fundamentada, estatisticamente validada e praticamente acionável.

### 6.1 Principais Contribuições

**Contribuição 1 — Evidência contraintuitiva com significância estatística e robustez a desbalanceamento**: Prompts simples (nome de classe único) superam formulações elaboradas em todas as lentes de avaliação: micro-average (0,527 [IC 95%: 0,515–0,539]), macro-average (0,506 [0,489–0,522]) e avaliação balanceada por classe+instância (0,513 [0,477–0,547]). Em micro-average, o ganho chega a 2,1–2,6×; em macro/balanceado, permanece em 1,4–1,7×. A significância estatística foi confirmada por testes de Wilcoxon pareados por imagem ($p < 10^{-54}$) com tamanho de efeito médio a grande (Cliff's $\delta$ = 0,43–0,52).

**Contribuição 2 — Diagnóstico de bottleneck por análise condicional**: A decomposição estágio-a-estágio revelou que 73,8% das falhas são grounding misses e apenas ~2,2% são falhas de segmentação pura. Quando o grounding funciona (box IoU ≥ 0,75), o SAM 2.1 produz Mask IoU = 0,849 e Dice = 0,914 — independente da formulação. Esse diagnóstico redireciona o esforço de otimização: a fronteira de melhoria relevante está no text-conditioned grounding, não na segmentação visual.

**Contribuição 3 — Protocolo de avaliação em 3 camadas**: O diagnóstico por estágio Detection → Mask quality → Practical utility (Boa / Corrigível / Ruim) separa dois tipos de falha com custos distintos para o anotador (não-detecção vs. máscara imprecisa) e permite cálculo direto do ROI de pré-rotulagem por categoria. O protocolo é agnóstico aos modelos subjacentes e pode ser aplicado a qualquer pipeline VLM de pré-rotulagem.

### 6.2 Limitações e Trabalhos Futuros

Os resultados são obtidos sobre um único par de modelos (Grounding DINO Swin-B + SAM 2.1 Hiera Large), um único benchmark (COCO val2017), com thresholds fixos e sem validação por anotadores humanos das categorias de utilidade. A generalização para outros domínios, modelos ou configurações requer avaliação adicional.

Três direções prioritárias de trabalho futuro emergem dos resultados:

1. **Ajuste fino do estágio de grounding**: Classes com taxa de detecção abaixo de 50% (*apple*, *bicycle*, *chair*) têm ROI reduzido para pré-rotulagem zero-shot. Ajuste fino supervisionado com amostras de anotações existentes poderia elevar a taxa de detecção dessas classes acima do limiar de utilidade.

2. **Avaliação com modelos de grounding de segunda geração**: Modelos como Grounding DINO 1.5 e Florence-2 utilizam text encoders mais robustos. A sensibilidade desses modelos à formulação do prompt é uma pergunta aberta — especialmente se o efeito *less is more* se inverte com encoders generativos (T5).

3. **Validação cruzada e avaliação humana**: Replicação em segundo dataset (LVIS, ADE20K) e estudo de tempo com anotadores humanos para validar as faixas de utilidade prática como proxies de economia real de esforço.

### 6.3 Consideração Final

A contribuição central deste trabalho pode ser enunciada de forma precisa: *em pipelines de pré-rotulagem com Grounding DINO, a formulação do prompt é o fator com maior impacto na qualidade final, esse efeito se manifesta quase integralmente no estágio de grounding, e a direção é "less is more"*. Essa evidência — contraintuitiva, replicável, estatisticamente significativa, e explicável de forma plausível pela arquitetura do modelo — substitui suposições implícitas por orientação baseada em dados para uma das decisões mais frequentes em pipelines de anotação assistida por VLMs.

---

## Declarations

### Funding

No funding was received for conducting this study.

### Competing Interests

The authors declare no competing interests.

### Author Contributions

Conceptualization, methodology, implementation, experiments, data curation, formal analysis, and manuscript drafting were performed by the author team and reviewed jointly. All authors approved the final manuscript.

### Data Availability

The processed experimental artifacts supporting the conclusions of this study are available in the repository under platform/vlm-pipeline/results and platform/vlm-pipeline/docs/paper.

### Code Availability

The pipeline source code and analysis scripts are available in the repository under platform/vlm-pipeline/src and platform/vlm-pipeline/statistical_analysis.py.

### Ethics Approval

Not applicable.

### Consent to Participate

Not applicable.

### Consent for Publication

Not applicable.
