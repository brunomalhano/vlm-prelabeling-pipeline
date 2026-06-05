# Seção 5 — Discussão

> **Extensão-alvo**: ~1.200 palavras

Esta seção interpreta os resultados da Seção 4 à luz da literatura revisada, propõe explicações plausíveis para os achados contraintuitivos e discute as implicações práticas e limitações da pesquisa.

---

## 5.1 Explicações Plausíveis para o Efeito *Less Is More*

O resultado central — prompts simples superam consistentemente formulações elaboradas em micro-average, macro-average e avaliação balanceada por classe+instância (Tabela 4.2) — contraria a intuição de que mais contexto deveria auxiliar o modelo. Em micro-average, o ganho é de 2,1–2,6×; quando controlamos o desbalanceamento (macro/balanceado), o ganho permanece substancial (1,4–1,7×). Três mecanismos arquiteturais são propostos como **explicações plausíveis**, com a ressalva de que este estudo não analisa diretamente mapas de atenção nem pesos internos dos modelos; as inferências a seguir são arquiteturais, não demonstrações causais.

**Explicação 1: Diluição de atenção no BERT-base.** O Grounding DINO utiliza o BERT-base como text encoder. No mecanismo de atenção cruzada entre texto e imagem, os tokens do prompt participam individualmente da interação com as features visuais. Um prompt como `the dog in the image` produz seis tokens: [CLS], `the`, `dog`, `in`, `the`, `image`, [SEP]. O token semanticamente informativo (`dog`) compete com tokens funcionais (`the`, `in`, `image`) pelo peso de atenção. No prompt simples `dog`, o token relevante concentra virtualmente toda a atenção cross-modal. É plausível que esse efeito de diluição de atenção explique parte da degradação progressiva com o aumento do comprimento da formulação.

**Explicação 2: Distribuição de treinamento do Grounding DINO.** O pré-treinamento do Grounding DINO usa a convenção de concatenar nomes de categorias simples separados por ponto (`cat . dog . person .`). Formulações como `segment the dog` ou `object: dog` estão fora da distribuição de treinamento — o modelo foi otimizado para reconhecer nomes de classe simples como âncoras textuais. A introdução de verbos de instrução ou prefixos de tipo produz uma combinação de tokens para a qual o modelo possivelmente não desenvolveu representação robusta durante o treinamento.

**Explicação 3: Separação de papéis GDINO/SAM.** O Grounding DINO foi projetado como localizador, não como seguidor de instruções. Formulações imperativas (`segment the dog`) pressupõem capacidade de decomposição instrucional que não faz parte do design do modelo. Ao contrário de modelos como InstructBLIP ou GPT-4V que processam instruções como primitivas, o GDINO interpreta o prompt inteiramente como descrição visual — e `segment the` é uma descrição visual inválida.

Essas três explicações convergem e são mutuamente reforçadoras: qualquer formulação que adicione tokens além do nome da classe introduz plausivelmente (a) diluição de atenção, (b) distribuição shift, e (c) tokens semanticamente parasitas. A magnitude do efeito — 50–60% de queda no mIoU — é coerente com a combinação dos três fatores, embora a contribuição relativa de cada um não possa ser quantificada sem análise de representações internas.

---

## 5.2 Por que o Bottleneck é o Grounding?

Os dados do E4 mostram que 73,8% das falhas são grounding misses e apenas ~2,2% são falhas de segmentação com box correto. Essa assimetria reflete as propriedades arquiteturais dos dois modelos.

O Grounding DINO opera em regime zero-shot: ele deve mapear um nome de classe para uma localização visual sem nenhum ajuste fino na distribuição específica de COCO val2017. As limitações são as dos modelos de detecção zero-shot em geral: performance cai para objetos pequenos, ocluídos, com aparência atípica ou em aglomerações.

O SAM 2.1, por outro lado, opera em regime prompting: ele recebe coordenadas de bounding box precisas e usa seu decoder de máscara para expandir essas coordenadas em uma segmentação. A tarefa é fundamentalmente mais restrita — o modelo não precisa "encontrar" o objeto, apenas "delinear" um objeto já localizado. A robustez do SAM 2.1 (Mask IoU = 0,849 quando box IoU ≥ 0,75) confirma que, para bounding boxes adequados, a segmentação de instâncias é uma tarefa bem-resolvida por modelos fundacionais atuais.

Essa análise tem implicação direta: **melhorias no pipeline devem ser investidas no estágio de grounding**, não no de segmentação. Estratégias como ajuste fino do Grounding DINO em classes de baixo desempenho, uso de thresholds adaptativos por categoria, ou substituição por um detector especializado para objetos pequenos teriam impacto significativo. A substituição do SAM 2.1 por alternativas de segmentação seria ineficiente: o bottleneck não está nesse estágio.

---

## 5.3 Verificação das Hipóteses

**H1 (confirmada)**: *Prompts simples alcançam mIoU significativamente maior que formulações complexas.*

O prompt simples produz mIoU = 0,527 [IC 95%: 0,515–0,539] em micro-average, 0,506 [0,489–0,522] em macro-average, e 0,513 [0,477–0,547] em avaliação balanceada por classe+instância, permanecendo acima de todas as alternativas nos três cenários. Os testes de Wilcoxon pareados por imagem rejeitam $H_0$ em todas as comparações ($p < 10^{-54}$), com tamanho de efeito médio a grande (Cliff's $\delta$ = 0,43–0,52).

**H2 (confirmada)**: *O efeito da formulação concentra-se no estágio de grounding, não no de segmentação.*

A decomposição E3 + E4 mostra queda acentuada na taxa de detecção com formulações elaboradas (micro: 66,3% para 24,8–30,7%; macro: 62,7% para 36,8–44,5%), enquanto o Mask IoU condicional a box IoU ≥ 0,75 permanece estável (0,838–0,861 entre formulações). A degradação é quase integralmente explicada pela queda no grounding, não por qualidade inferior da máscara.

**H3 (confirmada)**: *O principal bottleneck do pipeline é o grounding miss.*

73,8% das falhas (11.601 de 15.724 instâncias com erro) são grounding misses (E4/E5). Esse resultado é estrutural — mesmo com prompts simples, a taxa de não-detecção é ~33,7%.

---

## 5.4 Implicações Práticas

Os resultados produzem orientações diretas e acionáveis para equipes que implantam pipelines de pré-rotulagem:

1. **Use prompts simples (nomes de classe únicos)**: A evidência empírica contradiz a intuição de que instruções mais detalhadas melhoram o desempenho. Ferramentas de pré-rotulagem devem pré-configurar o formato de prompt simples por padrão e documentar explicitamente que formulações elaboradas reduzem a qualidade.

2. **A qualidade do pipeline é determinada pelo grounding**: Equipes de operações de dados devem concentrar esforço de otimização no estágio de grounding — ajuste de threshold, expansão do conjunto de classes de pré-treinamento, ou uso de detectores especializados para categorias de baixo desempenho. O SAM 2.1 não é o gargalo.

3. **Triagem por categoria antes da implantação**: Classes com alta taxa de não-detecção (e.g., apple: 64,8%, chair: 50,0%) têm ROI reduzido para pré-rotulagem automática. A triagem de classes por taxa de utilidade esperada deve preceder a implantação em escala.

4. **Protocolo de avaliação em 3 camadas é reutilizável**: O diagnóstico por estágio (Detection → Mask quality → Practical utility) pode ser aplicado a qualquer pipeline VLM de pré-rotulagem, independente dos modelos específicos. A separação permite identificar o bottleneck com precisão.

---

## 5.5 Limitações

**Generalização de modelos**: Os resultados são específicos para Grounding DINO (Swin-B) + SAM 2.1 (Hiera Large). O efeito *less is more* está plausivelmente vinculado ao text encoder BERT-base do GDINO Swin-B — arquiteturas com encoders distintos podem apresentar sensibilidade diferente à formulação do prompt:

| Modelo | Text Encoder | Hipótese sobre formulação | Prioridade |
|--------|-------------|--------------------------|-----------|
| **GDINO 1.5 Edge** | Encoder ampliado, treinado em Grounding-20M | O efeito *less is more* persiste? | Alta |
| **Florence-2** | T5 seq-to-seq + DaViT; instruções como input nativo | O efeito se *inverte*? | Alta |
| **YOLO-World** | CLIP ViT-B/32 (contrastive, sentence-level) | O efeito é atenuado? | Média |

**Ausência de validação por anotadores humanos**: As categorias de utilidade prática (Boa/Corrigível/Ruim) baseiam-se em faixas de IoU definidas a priori como proxy operacional. Não foram realizadas medições de tempo com anotadores humanos para validar que as faixas correspondem a economia real de esforço.

**Dependência do inglês**: Todos os prompts foram formulados em inglês. O efeito da formulação em outros idiomas não foi avaliado — embora o BERT-base do Grounding DINO seja predominantemente anglófono, modelos com encoders multilíngues podem apresentar comportamento distinto.

**Threshold fixo**: Os parâmetros `box_threshold=0.30` e `text_threshold=0.25` foram mantidos constantes. Uma análise de sensibilidade com thresholds adaptativos por formulação e por classe poderia alterar as conclusões quantitativas, embora a direção do efeito (simples > complexo) seja improvável de inverter dado o mecanismo arquitetural identificado.

**Cobertura de classes**: As 10 classes selecionadas do COCO cobrem objetos comuns mas não categorias especializadas (domínio médico, industrial, aéreo). A extrapolação para outros domínios deve ser feita com cautela.

**Dataset de validação único**: Todos os experimentos foram realizados em COCO val2017. Embora seja o benchmark mais utilizado na literatura de segmentação de instâncias, a validação em um segundo dataset (LVIS, ADE20K) fortaleceria a generalidade dos resultados.

**Formulações testadas**: As quatro formulações representam o espaço de variação mais comum na prática, mas não esgotam as possibilidades. Prompts com atributos visuais (`large dog`, `red car`) ou âncoras espaciais (`dog on the left`) não foram avaliados e poderiam apresentar comportamento distinto.
