# Seção 2 — Trabalhos Relacionados

> **Extensão-alvo**: ~1.500 palavras

---

## 2. Trabalhos Relacionados

Este trabalho situa-se na interseção de cinco linhas de pesquisa: segmentação open-vocabulary, grounding textual, segmentação promptável, anotação assistida e modelos visão-linguagem multilíngues. A revisão a seguir sintetiza os avanços relevantes em cada eixo e identifica a lacuna que motiva a presente investigação.

### 2.1 Segmentação Open-Vocabulary

A segmentação semântica tradicional opera sobre um conjunto fixo de classes definidas durante o treinamento. Modelos como U-Net, DeepLabV3+, Mask R-CNN e Mask2Former alcançam alto desempenho nesse regime, mas falham ao encontrar categorias não vistas durante o treino. A segmentação open-vocabulary busca superar essa limitação, permitindo segmentar objetos a partir de descrições textuais arbitrárias, inclusive para classes ausentes nos dados de treinamento.

Liang et al. [8] propuseram o OVSeg, um método em duas etapas no qual propostas de máscaras são geradas independentemente da classe e, em seguida, classificadas por um CLIP adaptado para imagens mascaradas. Os autores demonstraram que o CLIP pré-treinado em imagens naturais sofre degradação ao processar regiões mascaradas, alcançando apenas 20,1% de mIoU no ADE20K-150 com máscaras perfeitas — enquanto um classificador oráculo com propostas imperfeitas atinge 66,5%. A solução proposta, *mask prompt tuning*, reduz esse desvio de distribuição ao substituir regiões em branco por tokens visuais aprendíveis.

Revisões recentes de detecção open-vocabulary em imagens aéreas de UAV mapeiam a evolução do campo e identificam desafios específicos de escala e resolução [7].

**Observação crítica**: Os métodos de segmentação open-vocabulary avançaram significativamente na adaptação de modelos visão-linguagem para alinhar texto e regiões visuais. Entretanto, a avaliação nesses trabalhos assume o nome da classe como entrada textual canônica, sem investigar se formulações alternativas de prompt afetam a qualidade do alinhamento texto-imagem na etapa de grounding.

### 2.2 Grounding Textual e Detecção Open-Set

O grounding textual consiste em localizar objetos em uma imagem a partir de expressões em linguagem natural. Diferentemente da detecção de objetos convencional, que opera sobre categorias predefinidas, o grounding textual aceita descrições abertas — desde nomes de classes até expressões de referência complexas.

O Grounding DINO [2] combina um detector baseado em Transformer (DINO) com pré-treinamento grounded, aceitando entradas textuais como nomes de categorias ou expressões descritivas e retornando bounding boxes com scores de confiança. O modelo demonstrou desempenho competitivo em detecção zero-shot, sendo amplamente adotado como componente de localização em pipelines modulares.

Evidências aplicadas em cenários reais (mobilidade, tráfego e inspeção visual) reforçam que grounding textual é tecnicamente viável fora de benchmarks acadêmicos, mas ainda com alta sensibilidade a configuração de prompt e domínio.

**Observação crítica**: O Grounding DINO utiliza um text encoder baseado em BERT, treinado predominantemente com dados em inglês. O modelo foi avaliado na literatura exclusivamente com nomes de classe simples ou expressões curtas de referência. **Não há evidência sistemática sobre como a formulação do prompt — comprimento, estrutura, adição de palavras funcionais — afeta a qualidade do grounding**, representando uma lacuna diretamente relevante para a implantação de pipelines de pré-rotulagem.

### 2.3 Segmentação Promptável

O Segment Anything Model (SAM), proposto em [3], introduziu o paradigma de segmentação promptável: um modelo fundacional treinado em mais de 1 bilhão de máscaras, capaz de gerar segmentações a partir de prompts visuais como pontos, bounding boxes ou máscaras iniciais. O SAM demonstrou capacidade de generalização notável, produzindo máscaras de alta qualidade em domínios não vistos durante o treinamento.

O SAM 2 [4] estendeu o modelo original para vídeo e melhorou a qualidade das máscaras em imagens estáticas, com menor quantidade de artefatos e melhor aderência a bordas finas. Em aplicações especializadas, a qualidade final permanece dependente da precisão dos prompts visuais de entrada.

**Observação crítica**: A qualidade da máscara gerada pelo SAM depende diretamente da qualidade do prompt visual fornecido — tipicamente uma bounding box. Em pipelines modulares, a box é produzida por um modelo de grounding textual. Portanto, erros na etapa de grounding propagam-se para a segmentação. Essa dependência motiva a avaliação em duas camadas proposta neste trabalho.

### 2.4 Arquitetura dos Modelos Fundacionais Utilizados

O pipeline de pré-rotulagem avaliado neste trabalho é composto por dois modelos fundacionais com papéis complementares: o Grounding DINO para localização textual e o SAM 2.1 para geração de máscaras. Esta subseção descreve a arquitetura de cada modelo com o nível de detalhe necessário para compreender suas capacidades e limitações nos experimentos conduzidos.

#### 2.4.1 Grounding DINO (Swin-B)

O Grounding DINO [2] é um detector de objetos open-set com pré-treinamento grounded, aceitando descrições em linguagem natural como entrada para localizar objetos arbitrários. A arquitetura segue um paradigma *dual-encoder-single-decoder*, composto por cinco módulos principais: (i) backbone de imagem, (ii) backbone de texto, (iii) feature enhancer cross-modal, (iv) seleção de queries guiada por linguagem e (v) decoder cross-modal.

**Extração de features.** O backbone de imagem utiliza Swin-B, que processa a imagem e produz features multi-escala em diferentes estágios da rede. O backbone de texto emprega BERT-base para tokenização e embeddings contextuais. Essa escolha implica que tokens adicionais introduzidos por formulações complexas ("segment the", "object:", "in the image") também participam da atenção cruzada com as features visuais — potencialmente diluindo o sinal do nome da classe.

**Feature enhancer cross-modal.** Após a extração, as features de imagem e texto passam por um módulo de fusão composto por seis camadas de *feature enhancer*. Cada camada aplica: (a) self-attention deformável sobre as features de imagem; (b) self-attention sobre as features de texto; e (c) atenção cruzada bidirecional — image-to-text e text-to-image — para alinhar as representações das duas modalidades.

**Seleção de queries guiada por linguagem.** Em vez de usar queries estáticas como no DINO original, o Grounding DINO seleciona as $N_q = 900$ queries do decoder a partir das features de imagem com base na similaridade cross-modal com as features de texto. Dado um conjunto de features de imagem $\mathbf{X}_I \in \mathbb{R}^{N_I \times d}$ e features de texto $\mathbf{X}_T \in \mathbb{R}^{N_T \times d}$, os índices das top-$N_q$ queries são selecionados por:

$$\mathbf{I}_{N_q} = \text{Top}_{N_q}\left(\text{Max}_{(-1)}\left(\mathbf{X}_I \mathbf{X}_T^\top\right)\right)$$

Esse mecanismo garante que as queries iniciais do decoder já estejam semanticamente alinhadas com o prompt textual fornecido, direcionando a atenção do modelo para regiões da imagem relevantes ao texto.

**Decoder cross-modal.** O decoder é composto por seis camadas, cada uma contendo: self-attention sobre as queries, cross-attention para features de imagem (via atenção deformável), cross-attention para features de texto, e uma camada FFN. A camada adicional de text cross-attention em relação ao decoder do DINO original é responsável por manter o alinhamento linguístico durante o refinamento iterativo das predições. A saída final produz bounding boxes e scores de confiança por contrastive loss entre as queries de saída e os tokens de texto.

**Representação textual sub-sentença.** Para mitigar a interferência mútua entre nomes de categorias não relacionados quando concatenados em uma única sentença (prática comum em detecção open-vocabulary), o Grounding DINO introduz máscaras de atenção que bloqueiam a interação entre tokens de categorias distintas. Essa representação em nível de *sub-sentença* preserva features word-level para compreensão granular sem introduzir dependências espúrias entre categorias.

**Especificações da variante utilizada.** Neste trabalho, foi utilizada a variante Grounding DINO com backbone Swin-B (Base), que representa a configuração de maior capacidade totalmente open-source com pesos pré-treinados disponíveis publicamente. Os pesos foram pré-treinados em datasets de detecção (Objects365), grounding (GoldG) e captioning (Cap4M), garantindo ampla cobertura de conceitos visuais.

#### 2.4.2 SAM 2.1 (Hiera Large)

O Segment Anything Model 2 (SAM 2), proposto em [4], é um modelo fundacional para segmentação visual promptável que generaliza o SAM original [3] para o domínio de vídeo, mantendo compatibilidade com imagens estáticas. Quando aplicado a imagens individuais, o banco de memória permanece vazio e o modelo opera de forma análoga ao SAM original, porém com uma arquitetura interna significativamente reformulada. A versão 2.1 incorpora melhorias incrementais em qualidade de máscara e estabilidade de treinamento.

**Image encoder: Hiera.** Em contraste com o SAM original, que utiliza ViT-H como image encoder, o SAM 2.1 adota um encoder hierárquico (Hiera) pré-treinado com auto-supervisão. A arquitetura produz features em múltiplas escalas espaciais (estágios 1–4). Na variante Large (Hiera-L), utilizada neste trabalho, atenção global é aplicada em blocos específicos e os demais usam atenção em janelas (*windowed attention*) para eficiência computacional. As features dos estágios 3 e 4 são fundidas via FPN para gerar embeddings de imagem, enquanto features de alta resolução dos estágios 1 e 2 entram como *skip connections* para melhorar bordas finas.

**Prompt encoder.** O prompt encoder é idêntico ao do SAM original e aceita três tipos de prompt: (a) cliques positivos e negativos, representados por codificações posicionais somadas a embeddings aprendidos por tipo de prompt; (b) bounding boxes, codificadas como pares de pontos (canto superior esquerdo e canto inferior direito); e (c) máscaras, embutidas via convoluções e somadas ao embedding do frame. No pipeline deste trabalho, as bounding boxes produzidas pelo Grounding DINO são utilizadas como prompts do tipo (b).

**Mask decoder.** O decoder de máscara segue a arquitetura do SAM com blocos Transformer *two-way* que atualizam simultaneamente os embeddings de prompt e de frame. O SAM 2.1 adiciona duas extensões relevantes: (i) um head de predição de oclusão, que indica se o objeto de interesse está visível no frame atual (relevante para vídeo, mas inativo para imagens isoladas); e (ii) skip connections das features hierárquicas do image encoder (estágios 1 e 2) nas camadas de upsampling, que fornecem informação de alta resolução para delineamento preciso de bordas. Para prompts ambíguos (e.g., um único clique), o decoder prediz múltiplas máscaras candidatas com scores de IoU estimados, selecionando a de maior confiança.

**Módulos de memória.** Embora projetados para processamento de vídeo em streaming, os módulos de memória do SAM 2.1 merecem menção por completude arquitetural. O *memory attention* consiste em $L = 4$ blocos Transformer com self-attention, cross-attention para memórias espaciais e object pointers, seguidos de MLP. Utiliza 2D Rotary Positional Embedding (RoPE) para codificação espacial. O *memory encoder* gera representações de memória combinando a máscara predita com embeddings do image encoder via camadas convolucionais. O *memory bank* mantém uma fila FIFO de $N$ frames recentes e $M$ frames promovidos. Para segmentação de imagens estáticas — cenário deste trabalho — esses módulos não são ativados, e o modelo processa cada imagem de forma independente.

**Especificações da variante utilizada.** A variante Hiera-L opera a 30,2 FPS em GPU A100 para vídeo (batch size 1) e a 130,1 FPS para imagens (batch size 10). No benchmark SA-23 para segmentação de imagens, o SAM 2 (Hiera-B+) alcança 58,9% de mIoU com 1 clique, superando o SAM original (ViT-H, 58,1%) enquanto é 6× mais rápido. A variante Hiera-L atinge 59,5% de mIoU com ganhos adicionais de qualidade. Neste trabalho, a escolha da variante Hiera Large visa maximizar a qualidade da máscara gerada para isolar o efeito do grounding textual como variável experimental.

**Observação crítica**: O Grounding DINO e o SAM 2.1 operam em regimes complementares — o primeiro depende do text encoder (BERT-base) para interpretar o prompt textual, enquanto o segundo é agnóstico ao texto, recebendo apenas prompts visuais (bounding boxes). Essa separação de responsabilidades é central para o design experimental deste trabalho: permite diagnosticar se variações de qualidade causadas por diferentes formulações de prompt originam-se na etapa de compreensão linguística (grounding) ou na etapa puramente visual (segmentação).

### 2.5 Anotação Assistida e Pré-Rotulagem Automática

A construção de datasets rotulados permanece como um dos principais gargalos em projetos de visão computacional. A segmentação é particularmente custosa porque exige delimitação precisa de objetos em nível de pixel, com tempo de anotação significativamente superior ao de bounding boxes. O dataset COCO, amplamente utilizado como benchmark, evidencia que qualidade de rótulo é um fator determinante de desempenho downstream.

A abordagem de *human-in-the-loop* reposiciona o especialista humano como validador e corretor, em vez de criador de rótulos. Ferramentas como Polygon-RNN++ [9] já buscavam acelerar a criação de máscaras via anotação interativa. No contexto diretamente alinhado à segmentação por instâncias com muitas ocorrências por imagem, o estudo "Comparison of Manual and AI-assisted Labeling Techniques in Pixel-wise Instance Segmentation" [6] compara três regimes — manual, assistido interativo e assistido com pré-rotulagem — e reporta ganhos de produtividade com manutenção de qualidade em métricas como IoU, precisão, recall e F1.

Ganguly et al. [5] propuseram o Labeling Copilot, um agente de pesquisa profunda para curadoria automatizada de dados em visão computacional. O sistema orquestra três capacidades — descoberta calibrada, síntese controlada e anotação por consenso — utilizando um ensemble de modelos (DETIC, Grounding DINO e OWL-ViT) com mecanismo de fusão baseado em NMS e votação. No COCO, o módulo de anotação por consenso gera em média 14,2 propostas por imagem (quase o dobro das 7,4 do ground truth), alcançando 37,1% de mAP. O trabalho demonstra que pipelines automatizados podem produzir pseudo-labels robustos em escala industrial.

**Observação crítica**: O Labeling Copilot visa anotação autônoma com ensemble multi-modelo e mecanismo de consenso. Já o estudo IEEE (doc. 10883307) foca comparação de regimes de anotação assistida em segmentação densa. O presente trabalho tem objetivo distinto e complementar: avaliar um pipeline modular *simples* (dois modelos, sem consenso) como mecanismo de *pré-rotulagem para validação humana* e, principalmente, medir o efeito da **formulação dos prompts textuais** na qualidade dos pré-rótulos gerados.

### 2.6 Engenharia de Prompts para Modelos Visão-Linguagem

A qualidade dos prompts textuais é reconhecida como determinante no desempenho de modelos generativos. Em modelos visão-linguagem, a questão é análoga porém menos estudada: como a formulação textual afeta o alinhamento texto-imagem e, por consequência, tarefas downstream como retrieval, classificação e localização?

No contexto do CLIP, Radford et al. [10] demonstraram que o uso de templates como "a photo of a {class}" supera consistentemente o uso de nomes de classe isolados em classificação zero-shot, com ganho médio de 3.5% em ImageNet. Esse resultado motivou trabalhos subsequentes sobre aprendizado de prompts visuais (*prompt tuning*): CoOp [11] aprende um vetor de contexto contínuo que substitui o template manual, superando templates fixos em 11 dos 11 datasets avaliados. ProDA [12] estende esse paradigma com distribuições de prompts para capturar variabilidade semântica intra-classe.

Para tarefas de *grounding* e detecção, a literatura é escassa. Zhong et al. [13] propuseram o RegionCLIP, que adapta o CLIP para alinhamento de regiões em vez de imagens inteiras, e documentam que a qualidade dos templates de região impacta significativamente o zero-shot recognition. O Grounding DINO [2] utiliza como convenção a concatenação de nomes de classe separados por ponto ("cat . dog . bird ."), formato específico desenvolvido durante o pré-treinamento; os autores não avaliam sistematicamente outros formatos de prompt. O APE [14] demonstra que prompts de referência contextualizados ("the {class} in the center of the image") melhoram o grounding em benchmarks de referência, mas para objetos com âncora espacial explícita — um cenário distinto do usado em pré-rotulagem.

A sensibilidade de modelos de grounding a formulações de prompt permanece não quantificada para o caso de uso de pré-rotulagem: um único nome de classe como prompt, prompts com verbos de instrução ("segment the {class}"), prompts contextuais ("the {class} in the image") ou prompts com prefixo de objeto ("object: {class}"). Essa lacuna tem consequência prática direta: praticantes que implantam pipelines Grounding DINO + SAM escolhem um formato de prompt sem fundamentação empírica.

**Observação crítica**: A literatura de engenharia de prompts para VLMs estabeleceu que a formulação importa para classificação (CLIP) e que prompts contextuais com âncora espacial ajudam em referência (APE). Porém, **há pouca evidência sistemática sobre o impacto de diferentes formulações de prompt em um pipeline de grounding + segmentação para pré-rotulagem** — cenário de implantação mais comum para Grounding DINO + SAM 2.1. Os trabalhos existentes adotam nomes de classe simples como convenção de benchmarking sem justificação empírica explícita. Esta é a lacuna que o presente trabalho endereça.
