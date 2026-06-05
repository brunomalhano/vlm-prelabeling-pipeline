# Seção 1 — Introdução

> **Extensão-alvo**: ~1.000 palavras  
> **Estrutura**: SCQA (Situation–Complication–Question–Answer)

---

## 1. Introdução

### [S — Situação]

O desenvolvimento de sistemas de visão computacional depende fundamentalmente da disponibilidade de datasets rotulados de alta qualidade. Em tarefas de classificação, a anotação consiste na atribuição de uma classe a uma imagem inteira. Em detecção de objetos, exige a marcação de bounding boxes. Em segmentação — semântica, por instância ou panóptica — a anotação requer a delimitação precisa de cada objeto ou região em nível de pixel, o que torna o processo substancialmente mais caro, demorado e suscetível a inconsistências entre anotadores [1]. O próprio COCO — um dos benchmarks mais utilizados na área — tem sido continuamente revisado e refinado, reforçando que qualidade de anotação é um problema central, não um detalhe operacional.

Nesse contexto, os modelos visão-linguagem (VLMs) fundacionais abriram uma nova perspectiva para pré-rotulagem automática. O Grounding DINO [2] localiza objetos arbitrários em imagens a partir de descrições textuais, enquanto o Segment Anything Model [3] gera máscaras de segmentação a partir de prompts visuais como bounding boxes. A combinação desses modelos em um pipeline modular — grounding textual seguido de segmentação promptável — possibilita a geração automática de máscaras candidatas para qualquer classe descritível em linguagem natural, sem retreinamento.

### [C — Complicação]

Trabalhos recentes demonstram o potencial dessa abordagem. Ganguly et al. [5] propuseram o Labeling Copilot, um sistema agêntico que orquestra múltiplos modelos — incluindo Grounding DINO — para anotação automatizada em escala industrial. Em linha complementar, o estudo "Comparison of Manual and AI-assisted Labeling Techniques in Pixel-wise Instance Segmentation" [6] mostra que fluxos com apoio de IA e pré-rotulagem aumentam eficiência sem perda relevante de qualidade em cenários com muitas instâncias por imagem.

Entretanto, há uma decisão prática que todo praticante deve tomar ao implantar esses pipelines e que permanece sem resposta empírica na literatura: **qual formulação de prompt usar?** A literatura de open-vocabulary detection emprega sistematicamente nomes de classe simples — "dog", "car", "bottle" — como prompts para o Grounding DINO [2, 7]. Essa escolha raramente é justificada empiricamente; ela é herdada das convenções de benchmarking, não derivada de evidência experimental.

A intuição contrária é plausível: prompts mais descritivos deveriam fornecer contexto semântico adicional ao modelo. "The dog in the image" é mais específico do que "dog"; "segment the car" explicita a intenção; "object: bottle" alinha-se com a sintaxe de detecção. Se essa intuição estiver correta, pipelines de pré-rotulagem industriais estariam deixando desempenho na mesa ao usar apenas o nome da classe. Se estiver errada — e prompts complexos degradam a qualidade — os praticantes precisam saber por quê.

Essa questão tem implicações diretas para a implantação de pipelines de pré-rotulagem em escala. A escolha do formato de prompt afeta todos os objetos, todas as imagens e todas as iterações de anotação. Erros sistemáticos de formulação se propagam para a qualidade do dataset final.

### [Q — Questão]

Surge assim a pergunta central: **a formulação do prompt afeta significativamente a qualidade da pré-rotulagem gerada por um pipeline modular Grounding DINO + SAM 2.1?** Mais especificamente: o efeito se concentra na etapa de localização textual (grounding: text-to-box) ou na geração da máscara (segmentação: box-to-mask)? E qual é o principal modo de falha do pipeline — erros de detecção ou erros de segmentação?

### [A — Resposta / Contribuição]

Este trabalho apresenta uma avaliação sistemática de quatro formulações de prompt — simples, direta, contextual e orientada a objeto — no pipeline Grounding DINO (Swin-B) + SAM 2.1 (Hiera Large), aplicado ao COCO val2017 em 10 categorias de objetos. A avaliação é conduzida em três camadas complementares — localização (Box IoU, taxa de detecção), segmentação (Mask IoU, Dice, Boundary F1) e utilidade prática (classificação boa/corrigível/ruim) — permitindo diagnosticar a origem dos efeitos observados.

Os resultados revelam que prompts simples superam sistematicamente formulações complexas por margem substancial, e que o efeito concentra-se inteiramente na etapa de grounding: quando a localização textual é bem-sucedida, o SAM 2.1 entrega qualidade de máscara elevada independentemente da formulação original. A taxonomia de falhas identifica o grounding miss como modo de falha dominante do pipeline.

As contribuições do trabalho são:

1. **Evidência empírica contraintuitiva sobre formulação de prompts**: demonstração de que prompts simples (*single-word*) superam consistentemente formulações complexas em pipelines de grounding + segmentação, com significância estatística confirmada por bootstrap e teste de Wilcoxon pareado — resultado diretamente acionável por praticantes.

2. **Diagnóstico de origem de erros via análise condicional**: isolamento do impacto da etapa de grounding (text-to-box) da etapa de segmentação (box-to-mask), identificando que a degradação de qualidade é quase inteiramente determinada pela etapa de localização textual.

3. **Protocolo de avaliação em três camadas para pipelines de pré-rotulagem**: métricas de localização, segmentação e utilidade prática como protocolo reutilizável, agnóstico aos modelos subjacentes, para avaliar e comparar pipelines de pré-rotulagem baseados em VLMs.

O restante deste artigo está organizado da seguinte forma: a Seção 2 revisa os trabalhos relacionados nos cinco eixos que fundamentam a pesquisa; a Seção 3 descreve a metodologia experimental; a Seção 4 apresenta os resultados; a Seção 5 discute os achados; e a Seção 6 conclui o trabalho.
