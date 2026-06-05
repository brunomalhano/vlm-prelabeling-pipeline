# Pipeline de Pré-Rotulagem VLM — Relatório de Execução

> **Projeto**: Avaliação de prompts PT/EN para pré-rotulagem com Grounding DINO + SAM 2.1  
> **Data de Execução**: 14–15 de maio de 2026  
> **Autor**: Bruno Malhano (GBB Advisory)  
> **Status**: Concluído com sucesso

---

## 1. Resumo Executivo

Este documento descreve o processo completo de projeto, implementação, implantação e execução de um pipeline de pré-rotulagem baseado em Modelos de Linguagem-Visão (VLM) no Azure Container Apps com aceleração GPU. O pipeline avalia a eficácia do Grounding DINO + SAM 2.1 na geração de máscaras de segmentação a partir de prompts textuais em inglês e português, utilizando o COCO val2017 como dataset de referência.

**Descoberta Principal**: Prompts em inglês superam significativamente os prompts em português em todas as 10 categorias de objetos (delta médio de mIoU +0,19). O principal gargalo é a precisão do grounding (75,1% das falhas são detecções não realizadas), enquanto o estágio de segmentação apresenta bom desempenho quando o grounding é bem-sucedido (mIoU de máscara 0,84 com box IoU ≥ 0,5).

---

## 2. Desenho Experimental

### 2.1 Questões de Pesquisa

| ID | Questão |
|----|---------|
| **E1** | Qual é a qualidade baseline da pré-rotulagem VLM por categoria de objeto? |
| **E2** | O idioma do prompt (EN vs PT) afeta significativamente a qualidade da pré-rotulagem? |
| **E3** | Como o tipo de prompt interage com o idioma para influenciar a qualidade? |
| **E4** | O gargalo está no grounding (detecção) ou na segmentação (qualidade da máscara)? |
| **E5** | Quais são os modos de falha dominantes e como se distribuem? |

### 2.2 Dataset

- **Fonte**: COCO val2017 (5.000 imagens, 36.781 anotações)
- **Amostra**: 500 imagens via amostragem estratificada (seed=42), garantindo ≥30 instâncias por categoria
- **Categorias** (10):

| COCO ID | EN | PT | Artigo |
|---------|----|----|--------|
| 1 | person | pessoa | a |
| 2 | bicycle | bicicleta | a |
| 3 | car | carro | o |
| 17 | cat | gato | o |
| 18 | dog | cachorro | o |
| 44 | bottle | garrafa | a |
| 47 | cup | copo | o |
| 53 | apple | maçã | a |
| 59 | pizza | pizza | a |
| 62 | chair | cadeira | a |

### 2.3 Matriz de Prompts

4 tipos de prompt × 2 idiomas = **8 prompts por classe**, **80 prompts no total**:

| Tipo | Template em Inglês | Template em Português |
|------|-------------------|----------------------|
| **simple** | `{class}` | `{class_pt}` |
| **direct** | `segment the {class}` | `segmentar {artigo} {class_pt}` |
| **contextual** | `the {class} in the image` | `{artigo} {class_pt} na imagem` |
| **object** | `object: {class}` | `objeto: {class_pt}` |

### 2.4 Modelos

| Modelo | Arquitetura | Pesos | Papel |
|--------|------------|-------|-------|
| **Grounding DINO** | Swin-B backbone | `groundingdino_swinb_cogcoor.pth` (938 MB) | Detecção de objetos condicionada por texto (bounding boxes) |
| **SAM 2.1** | Hiera Large | `sam2.1_hiera_large.pt` (898 MB) | Segmentação de instâncias por box-prompt (máscaras) |

**Parâmetros de inferência**:
- Limiar de box (box threshold): 0,30
- Limiar de texto (text threshold): 0,25
- Algoritmo de correspondência: Húngaro (limiar de IoU ≥ 0,10)

### 2.5 Métricas

O pipeline avalia a qualidade da pré-rotulagem em três camadas complementares: detecção, qualidade da máscara e utilidade prática. Cada camada responde a uma pergunta diferente sobre o desempenho do sistema.

| Camada | Métricas |
|--------|----------|
| **Detecção** | Taxa de detecção, Box IoU |
| **Qualidade da Máscara** | Mask IoU, Coeficiente Dice, Boundary F1 (tolerância=2px) |
| **Utilidade** | Classificação: Bom (mIoU ≥ 0,75), Corrigível (≥ 0,50), Ruim (< 0,50) |

#### 2.5.1 Camada de Detecção

Estas métricas avaliam se o modelo consegue **localizar** o objeto na imagem a partir do prompt textual:

- **Taxa de Detecção**: Proporção de objetos presentes no ground truth que foram detectados pelo Grounding DINO (ou seja, que receberam uma bounding box predita com correspondência via algoritmo Húngaro). Uma taxa de 86% para "person" significa que, de cada 100 pessoas anotadas no ground truth, o modelo encontrou 86. As 14 restantes são faltas de grounding (grounding misses).

- **Box IoU (Intersection over Union da Bounding Box)**: Mede a sobreposição entre a bounding box predita e a bounding box do ground truth. Calculada como:

$$\text{Box IoU} = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

  Um valor de 1,0 indica alinhamento perfeito; 0,0 indica nenhuma sobreposição. No pipeline, o Box IoU é usado para (a) correspondência Húngara entre predições e ground truth (limiar mínimo 0,10) e (b) diagnóstico no experimento E4 — separar erros de grounding (box IoU < 0,50) de erros de segmentação.

#### 2.5.2 Camada de Qualidade da Máscara

Estas métricas avaliam a **precisão pixel a pixel** da máscara de segmentação gerada pelo SAM 2.1 em comparação com a máscara do ground truth:

- **Mask IoU (Intersection over Union da Máscara)**: Extensão do IoU para máscaras binárias de segmentação. Calcula a razão entre a área de sobreposição e a área de união das máscaras predita e ground truth:

$$\text{Mask IoU} = \frac{\sum (M_{\text{pred}} \land M_{\text{gt}})}{\sum (M_{\text{pred}} \lor M_{\text{gt}})}$$

  Onde $M_{\text{pred}}$ e $M_{\text{gt}}$ são matrizes binárias (0 ou 1) com o mesmo tamanho da imagem. É a **métrica principal do pipeline** — usada para classificação de utilidade e para os deltas nos experimentos E2 e E3. Valores típicos: 0,84 indica máscara excelente, 0,50 indica máscara parcialmente útil, <0,30 indica máscara de baixa qualidade.

- **Coeficiente Dice (F1 Score a nível de pixel)**: Também mede a sobreposição de máscaras, porém é mais sensível a objetos pequenos por não penalizar tão severamente a área de união:

$$\text{Dice} = \frac{2 |M_{\text{pred}} \cap M_{\text{gt}}|}{|M_{\text{pred}}| + |M_{\text{gt}}|}$$

  O Dice é sempre ≥ Mask IoU para a mesma predição (ex.: Mask IoU de 0,686 corresponde a Dice de 0,723 para "cat"). Ele complementa o IoU ao fornecer uma perspectiva onde o equilíbrio entre precisão e recall é igualmente ponderado, sem o efeito de penalização dupla da união. Em termos práticos, Dice > 0,80 indica segmentação de qualidade clínica/industrial.

- **Boundary F1 (F1 de Contorno)**: Mede a qualidade dos **bordos** da máscara, independentemente da região interna. O cálculo segue três etapas:
  1. Extrai os pixels de contorno de ambas as máscaras (predita e ground truth) usando o algoritmo `find_contours` com nível 0,5
  2. Para cada pixel de contorno predito, verifica se existe um pixel de contorno do ground truth dentro de uma janela de **tolerância de 2 pixels** (distância de Chebyshev)
  3. Calcula a precisão (fração de contornos preditos que têm correspondente no GT), o recall (fração de contornos GT que têm correspondente na predição), e o F1 como média harmônica:

$$\text{BF1} = \frac{2 \cdot P_{\text{contorno}} \cdot R_{\text{contorno}}}{P_{\text{contorno}} + R_{\text{contorno}}}$$

  Esta métrica é particularmente importante para pré-rotulagem porque **bordas imprecisas são o principal custo de correção manual**. Uma máscara com IoU alto mas BF1 baixo (ex.: cat com IoU 0,686 mas BF1 0,521) indica que a região geral está correta, porém os bordos precisam de refinamento — o que ainda é significativamente menos trabalho que rotular do zero.

#### 2.5.3 Camada de Utilidade Prática

A classificação de utilidade traduz as métricas numéricas em **decisões operacionais** para o fluxo de rotulagem. Cada instância segmentada é classificada com base no seu Mask IoU:

| Classificação | Limiar de mIoU | Significado Operacional |
|--------------|:--------------:|------------------------|
| **Bom** | ≥ 0,75 | Máscara utilizável diretamente ou com ajustes mínimos. Pode ser aceita sem revisão detalhada. Economia estimada: **90%+ do tempo de rotulagem manual**. |
| **Corrigível** | ≥ 0,50 e < 0,75 | Máscara captura a região geral do objeto mas requer correções de borda ou ajustes de cobertura. Fornece um ponto de partida útil. Economia estimada: **50-70% do tempo de rotulagem**. |
| **Ruim** | < 0,50 | Máscara incorreta, ausente ou cobrindo região errada. Deve ser descartada — é mais rápido rotular do zero do que corrigir. **Nenhuma economia de tempo.** |

Estes limiares (0,75 e 0,50) foram definidos com base em literatura de active learning e human-in-the-loop annotation, onde mIoU ≥ 0,75 é amplamente aceito como limiar de qualidade para pré-rotulagem automática em datasets de segmentação de instâncias.

#### 2.5.4 Algoritmo de Correspondência

Para calcular as métricas acima, é necessário primeiro determinar **qual predição corresponde a qual anotação do ground truth**. O pipeline utiliza o **algoritmo Húngaro** (implementado via `scipy.optimize.linear_sum_assignment`):

1. Constrói uma matriz de custo $C_{ij}$ onde cada elemento é $1 - \text{IoU}(pred_i, gt_j)$
2. Resolve o problema de atribuição ótima que minimiza o custo total (maximiza a soma de IoU)
3. Aplica um limiar mínimo de IoU ≥ 0,10 — pares com IoU inferior são descartados como não correspondentes

Predições sem correspondente no GT são contabilizadas como **falsos positivos**. Anotações GT sem correspondente são contabilizadas como **faltas de grounding** (grounding misses). Ambos alimentam o experimento E5 (taxonomia de falhas).

---

## 3. Arquitetura do Pipeline

### 3.1 Estrutura de Módulos

```
src/vlm_pipeline/
├── __init__.py          # Inicialização do pacote
├── __main__.py          # Ponto de entrada (python -m vlm_pipeline)
├── cli.py               # CLI: subcomandos sample, run, visualize
├── sampling.py          # Amostragem estratificada das anotações COCO
├── prompts.py           # Geração de prompts (10 classes × 4 tipos × 2 idiomas)
├── grounding.py         # Inferência Grounding DINO (texto → boxes)
├── segmentation.py      # Inferência SAM 2.1 (boxes → máscaras)
├── matching.py          # Correspondência Húngara (predições ↔ ground truth)
├── metrics.py           # IoU, Dice, Boundary F1, classificação de utilidade
├── pipeline.py          # Orquestrador: encadeia todos os módulos + gera tabelas E1–E5
└── visualization.py     # Geração de figuras (gráficos + exemplos de falha)
```

### 3.2 Fluxo de Processamento

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Por Imagem (×500)                              │
│                                                                     │
│  Para cada (classe × tipo_prompt × idioma):                         │
│                                                                     │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Gerador  │───▶│ Grounding     │───▶│ SAM 2.1  │───▶│  Corresp.│ │
│  │ de Prompt │    │ DINO (boxes)  │    │(máscaras)│    │ Húngara  │ │
│  └──────────┘    └───────────────┘    └──────────┘    └──────────┘ │
│                                                            │        │
│                                                     ┌──────▼──────┐ │
│                                                     │  Métricas   │ │
│                                                     │ IoU/Dice/BF1│ │
│                                                     └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Agregar Resultados │
                    │  Tabelas CSV E1–E5  │
                    │  + Exemplos de Falha│
                    └─────────────────────┘
```

Cada imagem produz um arquivo JSON com todos os resultados de instâncias em todas as combinações de prompt, totalizando **44.201 resultados de instâncias** a partir de 500 imagens.

---

## 4. Infraestrutura e Implantação

### 4.1 Recursos Azure

| Recurso | Nome | Região | Finalidade |
|---------|------|--------|------------|
| **Grupo de Recursos** | rg-trainning-models | West US | Contêiner para todos os recursos |
| **Ambiente Container Apps** | managedEnvironment-rgtrainningmode-8b63 | West US | Hospedagem ACA com perfis GPU |
| **Container Apps Job** | vlm-pipeline-job | West US | Execução de job com GPU |
| **Container Registry** | acrkvbqwbbjjkoy6 | East US 2 | Armazenamento de imagens Docker |
| **Storage Account** | stvlmpipe702226 | West US | Dados, pesos, resultados |

### 4.2 Perfil de Computação

| Especificação | Valor |
|---------------|-------|
| **Perfil de Workload** | NC8as-T4 |
| **vCPUs** | 8 |
| **Memória** | 56 GiB |
| **GPU** | 1× NVIDIA Tesla T4 (16 GB VRAM) |
| **Timeout da Réplica** | 21.600s (6 horas) |

### 4.3 Imagem de Contêiner

```dockerfile
# Base: NVIDIA CUDA 12.1.1 runtime no Ubuntu 22.04
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Python 3.10 + azcopy para acesso a blobs via identidade gerenciada
# PyTorch pré-instalado do índice cu121 para compatibilidade com driver CUDA 12.x
# transformers fixado em < 4.45.0 (compatibilidade com groundingdino)
```

**Estratégia de build**: Contexto de build limpo em `/tmp/vlm-build-context/` (~224 KB), pois o diretório de trabalho contém arquivos grandes (3,3 GB). Compilado via `az acr build` no Azure Container Registry.

### 4.4 Arquitetura de Armazenamento

| Contêiner | Conteúdo | Tamanho |
|-----------|----------|---------|
| **vlm-data** | Imagens COCO val2017 (5.000 JPEGs) + anotações | ~834 MB |
| **vlm-weights** | Pesos dos modelos Grounding DINO + SAM 2.1 | ~1,8 GB |
| **vlm-results** | JSONs, CSVs e PNGs de saída | ~25 MB |

**Modelo de acesso**: Identidade Gerenciada atribuída pelo sistema no ACA job com funções RBAC `Storage Blob Data Reader` + `Storage Blob Data Contributor`. Os dados são baixados na inicialização do contêiner via `azcopy` com `AZCOPY_AUTO_LOGIN_TYPE=MSI`.

### 4.5 Entrypoint do Job (`run_job.sh`)

```
[1/5] Baixar dados do Azure Blob        (azcopy + identidade gerenciada)
[2/5] Validar dados                     (verificar imagens, anotações, pesos)
[3/5] Gerar amostra estratificada       (python -m vlm_pipeline sample)
[4/5] Executar pipeline de experimentos  (python -m vlm_pipeline run)
[5/5] Gerar visualizações               (python -m vlm_pipeline visualize)
  └── Upload de resultados para Blob    (azcopy para vlm-results/run-{timestamp}/)
```

---

## 5. Desafios de Implantação e Resoluções

O pipeline exigiu 9 iterações de build até alcançar a execução bem-sucedida. A seguir, a cronologia dos problemas encontrados e resolvidos:

### 5.1 Linha do Tempo de Problemas

| Build | Execução | Problema | Causa Raiz | Resolução |
|-------|----------|----------|------------|-----------|
| ch2 | axcwdzj | **VolumeMountFailure** | Montagem SMB do Azure Files requer `allowSharedKeyAccess`, bloqueado por política organizacional | Removidas montagens de volume; migrado para download de blob via identidade gerenciada + azcopy |
| ch4 | cw0kpd2 | **ModuleNotFoundError** `vlm_pipeline.__main__` | Ponto de entrada `__main__.py` ausente | Criado `src/vlm_pipeline/__main__.py` |
| ch5 | ynpbak6 | **AttributeError** `BertModel.get_head_mask` | Versão incompatível do `transformers` com groundingdino | Fixada versão `transformers<4.45.0` no requirements.txt |
| ch6 | txt04v1 | **FileNotFoundError** para imagens | `azcopy --recursive` cria diretório aninhado `val2017/val2017/` | Adicionada correção automática no run_job.sh para detectar e desaninhar diretórios |
| ch7 | kn3e08y | Execução diagnóstica | Parada manual para verificar que downloads funcionavam | N/A |
| ch8 | je1dvsl | **Exit code 2** (pipefail) | `ls inexistente \| head` retorna não-zero, matando o script sob `set -euo pipefail` | Adicionado `\|\| true` a todos os comandos diagnósticos |
| ch9 | vhzzuuf | **Timeout em 472/500** (94%) | Timeout de réplica de 4 horas era insuficiente (~32s/imagem × 500 = 4h24m) | Aumentado `replicaTimeout` de 14.400s para 21.600s (6 horas) |
| ch9 | **547oszp** | **Sucesso** | Todos os problemas resolvidos | Pipeline concluído em 4h21m com margem de 1h39m |

### 5.2 Decisões Técnicas Principais

1. **Blob em vez de Azure Files**: A política organizacional bloqueava o acesso por chave compartilhada necessário para montagens SMB. Identidade gerenciada + azcopy forneceu uma alternativa mais limpa e segura.

2. **Pré-instalação PyTorch cu121**: A imagem base CUDA 12.1.1 requer PyTorch compilado para CUDA 12.1. A instalação pelo índice padrão do PyPI traria cu130, causando incompatibilidade de driver. Solução: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` explícito antes do `pip install -r requirements.txt`.

3. **Fixação da versão do transformers**: O groundingdino usa APIs internas (`BertModel.get_head_mask`) que foram removidas/renomeadas no transformers ≥ 4.45.0. Fixar em `<4.45.0` garante compatibilidade.

4. **Timeout de 6 horas**: A ~32s por imagem para 500 imagens, somente a inferência leva ~4h24m. Adicionando overhead de download, validação, amostragem, visualização e upload, 6 horas fornecem margem adequada.

---

## 6. Resumo da Execução

| Métrica | Valor |
|---------|-------|
| **ID da Execução** | vlm-pipeline-job-547oszp |
| **Hora de Início** | 2026-05-14 20:51:08 UTC |
| **Hora de Término** | 2026-05-15 01:12:34 UTC |
| **Tempo Total** | 4 horas e 21 minutos |
| **Imagens Processadas** | 500 / 500 (100%) |
| **Resultados de Instâncias** | 44.201 |
| **Velocidade Média** | ~31,4s por imagem |
| **Arquivos Enviados** | 521 (500 JSONs brutos + 8 CSVs + 13 PNGs) |
| **Localização dos Resultados** | `vlm-results/run-20260515T011220Z` |

---

## 7. Resultados

### 7.1 E1 — Qualidade Baseline (Prompts Simples em EN)

Baseline usando o tipo de prompt mais simples ("simple") em inglês nas 500 imagens:

| Categoria | mIoU Média | Dice Médio | BF1 Médio | Taxa de Detecção | Instâncias |
|-----------|:----------:|:----------:|:---------:|:----------------:|:----------:|
| **cat** | 0,686 | 0,723 | 0,521 | 93,8% | 65 |
| **dog** | 0,668 | 0,713 | 0,549 | 90,9% | 77 |
| **person** | 0,544 | 0,598 | 0,505 | 86,1% | 2.284 |
| **pizza** | 0,457 | 0,495 | 0,306 | 62,7% | 161 |
| **bottle** | 0,429 | 0,467 | 0,460 | 75,4% | 696 |
| **car** | 0,422 | 0,458 | 0,420 | 62,3% | 464 |
| **cup** | 0,342 | 0,364 | 0,336 | 47,6% | 557 |
| **chair** | 0,297 | 0,345 | 0,281 | 56,0% | 820 |
| **bicycle** | 0,264 | 0,316 | 0,230 | 54,6% | 196 |
| **apple** | 0,231 | 0,258 | 0,198 | 42,4% | 118 |

**Distribuição de Utilidade** (% de instâncias por faixa de qualidade):

| Categoria | Bom (≥0,75) | Corrigível (≥0,50) | Ruim (<0,50) |
|-----------|:-----------:|:------------------:|:------------:|
| cat | 73,8% | 1,5% | 24,6% |
| dog | 70,1% | 5,2% | 24,7% |
| person | 51,6% | 12,9% | 35,5% |
| pizza | 44,7% | 6,8% | 48,4% |
| car | 42,2% | 7,3% | 50,4% |
| bottle | 41,7% | 8,8% | 49,6% |
| cup | 35,2% | 3,6% | 61,2% |
| apple | 21,2% | 4,2% | 74,6% |
| chair | 19,3% | 16,6% | 64,1% |
| bicycle | 12,8% | 19,4% | 67,9% |

**Interpretação**: Objetos grandes e bem definidos (cat, dog, person) alcançam qualidade forte de pré-rotulagem (>50% Bom). Objetos pequenos ou ocluídos (apple, chair, bicycle) produzem máscaras majoritariamente inutilizáveis (<20% Bom).

### 7.2 E2 — Comparação de Idiomas (EN vs PT)

Prompts em inglês superam consistentemente os prompts em português em todas as categorias:

| Categoria | mIoU EN | mIoU PT | Delta (EN−PT) | IC 95% Inferior | IC 95% Superior | Significativo? |
|-----------|:-------:|:-------:|:-------------:|:---------------:|:---------------:|:--------------:|
| **dog** | 0,630 | 0,295 | **+0,335** | 0,276 | 0,392 | Sim |
| **bottle** | 0,242 | 0,021 | **+0,222** | 0,206 | 0,237 | Sim |
| **cat** | 0,613 | 0,397 | **+0,215** | 0,138 | 0,283 | Sim |
| **chair** | 0,202 | 0,003 | **+0,199** | 0,187 | 0,210 | Sim |
| **cup** | 0,214 | 0,014 | **+0,200** | 0,186 | 0,216 | Sim |
| **car** | 0,232 | 0,039 | **+0,194** | 0,175 | 0,212 | Sim |
| **person** | 0,288 | 0,098 | **+0,191** | 0,180 | 0,201 | Sim |
| **bicycle** | 0,208 | 0,044 | **+0,163** | 0,138 | 0,189 | Sim |
| **apple** | 0,142 | 0,020 | **+0,122** | 0,093 | 0,152 | Sim |
| **pizza** | 0,388 | 0,355 | +0,033 | −0,011 | 0,076 | **Não** |

**Interpretação**: Prompts em inglês produzem máscaras de qualidade significativamente superior para 9 das 10 categorias (IC bootstrap 95% exclui zero). Apenas "pizza" não apresenta diferença significativa — provavelmente porque "pizza" é um cognato idêntico em ambos os idiomas. A vantagem média do inglês é de **+0,19 mIoU**.

### 7.3 E3 — Interação Tipo de Prompt × Idioma

| Tipo de Prompt | mIoU EN | mIoU PT | Dice EN | Dice PT |
|---------------|:-------:|:-------:|:-------:|:-------:|
| **simple** | **0,445** | 0,077 | **0,490** | 0,082 |
| contextual | 0,221 | 0,065 | 0,240 | 0,070 |
| direct | 0,190 | 0,066 | 0,206 | 0,070 |
| object | 0,180 | 0,061 | 0,196 | 0,065 |

**Interpretação**: O tipo de prompt "simple" (nome da classe sem decoração) é dramaticamente mais eficaz que todos os outros tipos em inglês (mIoU 0,445 vs ≤0,221). Em português, todos os tipos de prompt apresentam desempenho similarmente baixo (mIoU 0,061–0,077). Isso sugere que o codificador de texto do Grounding DINO responde melhor a nomes de classe limpos e sem adornos, e que sintaxe adicional ("segment the", "object:") introduz ruído em vez de contexto útil.

### 7.4 E4 — Diagnóstico Grounding vs Segmentação

**Qualidade condicional da máscara (quando o grounding é bem-sucedido)**:

| Limiar de Box IoU | Instâncias (n) | mIoU da Máscara | Dice Médio | BF1 Médio |
|:-----------------:|:--------------:|:---------------:|:----------:|:---------:|
| ≥ 0,50 | 8.230 | **0,836** | **0,905** | 0,695 |
| ≥ 0,75 | 7.456 | **0,857** | **0,919** | 0,706 |

**Distribuição da fonte de erros** (entre 16.084 detecções correspondidas):

| Tipo de Erro | Contagem | Percentual |
|-------------|:--------:|:----------:|
| Erros de grounding (box IoU < 0,50) | 7.854 | **48,8%** |
| Erros de segmentação (box IoU ≥ 0,50 mas mask IoU < 0,50) | 242 | **1,5%** |

**Interpretação**: Quando o Grounding DINO posiciona a bounding box com precisão (box IoU ≥ 0,50), o SAM 2.1 produz máscaras de segmentação excelentes (mIoU 0,84, Dice 0,90). Erros de grounding superam os erros de segmentação na proporção **32:1** (48,8% vs 1,5%). **O estágio de grounding é esmagadoramente o gargalo** — investir em detecção condicionada por texto mais precisa produziria o maior ganho de qualidade.

### 7.5 E5 — Taxonomia de Falhas

Distribuição dos modos de falha entre todos os resultados de instâncias que não atingiram a qualidade "Bom" (37.427 falhas):

| Tipo de Falha | Contagem | Percentual | Descrição |
|--------------|:--------:|:----------:|-----------|
| **grounding_miss** | 28.117 | **75,1%** | Objeto não detectado |
| **false_positive** | 7.481 | 20,0% | Objeto predito sem correspondente no GT |
| mask_incomplete | 1.312 | 3,5% | Box correto mas máscara subdimensionada |
| box_incomplete | 346 | 0,9% | Box detectado mas pequeno/deslocado demais |
| mask_excessive | 171 | 0,5% | Box correto mas máscara superdimensionada |

**Interpretação**: Três em cada quatro falhas (75,1%) são faltas completas de grounding — o modelo simplesmente não consegue detectar o objeto-alvo a partir do prompt textual. Isso corrobora a descoberta do E4 de que a qualidade da segmentação é forte quando a detecção é bem-sucedida. Falsos positivos (20%) indicam que o modelo também superdetecta em alguns casos. Erros no nível da máscara (incompleta/excessiva) representam apenas 4% de todas as falhas.

---

## 8. Saídas de Visualização

O pipeline gerou 13 arquivos de visualização:

| Arquivo | Descrição |
|---------|-----------|
| `figures/e1_class_comparison.png` | Gráfico de barras de mIoU/Dice/BF1 por classe |
| `figures/e2_lang_heatmap.png` | Mapa de calor de mIoU EN vs PT por classe |
| `figures/e3_prompt_interaction.png` | Gráfico de barras agrupadas de tipo de prompt × idioma |
| `figures/e4_box_vs_mask.png` | Gráfico de dispersão de box IoU vs mask IoU |
| `figures/e5_failure_taxonomy.png` | Gráfico de pizza/barras da distribuição de modos de falha |
| `figures/e5_failures/failure_0[0-7].png` | 8 estudos de caso de falha anotados |

---

## 9. Conclusões

1. **A pré-rotulagem VLM é viável para objetos grandes e bem definidos** — categorias como cat (73,8% Bom), dog (70,1% Bom) e person (51,6% Bom) podem se beneficiar deste pipeline como etapa de pré-anotação que reduz significativamente o esforço de rotulagem manual.

2. **Prompts em inglês são obrigatórios para uso em produção** — prompts em português produzem qualidade drasticamente inferior em 9/10 categorias, com deltas médios de mIoU de +0,19 a favor do inglês. O codificador de texto BERT do Grounding DINO foi treinado predominantemente com texto em inglês.

3. **Prompts simples superam prompts verbosos** — nomes de classe simples ("person", "car") produzem resultados 2× melhores que prompts contextuais ou diretivos. Sintaxe adicional degrada a precisão do grounding.

4. **O grounding é o gargalo, não a segmentação** — o SAM 2.1 alcança mIoU de máscara de 0,84 quando o grounding é preciso. Melhorar a precisão de detecção (ex.: modelos maiores do Grounding DINO, fine-tuning ou abordagens ensemble) produziria os maiores retornos.

5. **Objetos pequenos continuam desafiadores** — apple (21,2% Bom), chair (19,3% Bom) e bicycle (12,8% Bom) são mal atendidos por este pipeline. Tamanho do objeto e oclusão são os fatores limitantes primários.

---

## 10. Recomendações

| Prioridade | Ação | Impacto Esperado |
|-----------|------|------------------|
| **Alta** | Usar **prompts "simple" em inglês** exclusivamente em fluxos de pré-rotulagem em produção | +2× mIoU vs outros tipos de prompt |
| **Alta** | Implementar **filtragem baseada em confiança** — aceitar apenas pré-rótulos com confiança de detecção > 0,5 | Reduz taxa de falsos positivos de 20% |
| **Média** | Avaliar **Grounding DINO 1.5** ou **Grounding DINO 2** para melhor precisão de detecção | Endereça o gargalo de 75% de faltas de grounding |
| **Média** | Aplicar **limiares específicos por categoria** — limiar de box menor para objetos pequenos, maior para grandes | Melhora taxa de detecção para categorias desafiadoras |
| **Baixa** | Considerar **fine-tuning do Grounding DINO** com dados específicos de domínio em português | Reduz a lacuna EN/PT para domínios especializados |

---

## 11. Reprodutibilidade

### 11.1 Re-executando o Pipeline

```bash
# 1. Compilar a imagem (a partir de contexto limpo)
cp Dockerfile requirements.txt setup.py /tmp/vlm-build-context/
cp -r src/ configs/ scripts/ /tmp/vlm-build-context/
cd /tmp/vlm-build-context
az acr build --registry acrkvbqwbbjjkoy6 --image vlm-pipeline:latest --file Dockerfile .

# 2. Iniciar o job
az containerapp job start \
    --name vlm-pipeline-job \
    --resource-group rg-trainning-models

# 3. Monitorar progresso
az containerapp job execution list \
    --name vlm-pipeline-job \
    --resource-group rg-trainning-models \
    --query "[0].{name:name, status:properties.status}" -o table

# 4. Baixar resultados (habilitar temporariamente acesso público)
az storage account update -n stvlmpipe702226 --public-network-access Enabled
azcopy copy "https://stvlmpipe702226.blob.core.windows.net/vlm-results/run-*" ./results/ --recursive
az storage account update -n stvlmpipe702226 --public-network-access Disabled
```

### 11.2 Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `AZURE_STORAGE_ACCOUNT` | `stvlmpipe702226` | Nome da storage account |
| `DATA_CONTAINER` | `vlm-data` | Contêiner blob com dados COCO |
| `WEIGHTS_CONTAINER` | `vlm-weights` | Contêiner blob com pesos dos modelos |
| `RESULTS_CONTAINER` | `vlm-results` | Contêiner blob para saídas |

### 11.3 Configuração

Todos os parâmetros do experimento estão em `configs/experiment.yaml`:
- **Seed**: 42 (amostragem determinística)
- **Tamanho da amostra**: 500 imagens
- **Classes**: 10 categorias COCO
- **Prompts**: 4 tipos × 2 idiomas = 80 prompts
- **Grounding DINO**: box_threshold=0,30, text_threshold=0,25
- **Correspondência**: Algoritmo Húngaro, limiar de IoU=0,10
- **Limiares de utilidade**: Bom ≥ 0,75, Corrigível ≥ 0,50

### 11.4 Testes

O pipeline inclui 40 testes unitários cobrindo todos os módulos:

```bash
cd platform/vlm-pipeline
python -m pytest tests/ -v  # 40/40 passando
```

---

## 12. Inventário de Artefatos

### 12.1 Arquivos de Resultado (521 no total)

| Diretório | Arquivos | Descrição |
|-----------|:--------:|-----------|
| `results/raw/` | 500 | JSON por imagem com todos os resultados de instâncias |
| `results/tables/` | 8 | Tabelas CSV agregadas (E1–E5) |
| `results/figures/` | 5 | Gráficos dos experimentos (E1–E5) |
| `results/figures/e5_failures/` | 8 | Estudos de caso de falha anotados |

### 12.2 Arquivos de Tabela

| Arquivo | Conteúdo |
|---------|----------|
| `e1_overall.csv` | mIoU, Dice, BF1 e taxa de detecção por classe (baseline) |
| `e1_utility_distribution.csv` | Percentuais de Bom/Corrigível/Ruim por classe |
| `e2_lang_comparison.csv` | Métricas por classe × idioma |
| `e2_lang_delta.csv` | Delta EN−PT com IC bootstrap 95% |
| `e3_prompt_type.csv` | Métricas por tipo de prompt × idioma |
| `e4_conditional.csv` | Qualidade da máscara condicionada a limiares de box IoU |
| `e4_error_source.csv` | Distribuição de erros de grounding vs segmentação |
| `e5_failure_taxonomy.csv` | Distribuição dos modos de falha |

---

*Gerado a partir da execução `vlm-pipeline-job-547oszp` no Azure Container Apps (NC8as-T4, West US).*
