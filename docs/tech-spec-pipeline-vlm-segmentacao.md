# Especificação Técnica — Pipeline de Pré-Rotulagem VLM + Segmentação

> **Projeto**: Avaliação de prompts PT/EN para pré-rotulagem com Grounding DINO + SAM  
> **Data**: 2026-05-12  
> **Ambiente**: Cloud GPU (NVIDIA A100 40GB), PyTorch 2.x, Python 3.10+

---

## 1. Visão Geral da Arquitetura

```
projeto/
├── configs/
│   └── experiment.yaml          # Parâmetros do experimento (thresholds, classes, prompts)
├── data/
│   ├── coco/
│   │   ├── val2017/             # Imagens COCO val2017 (download automático)
│   │   └── annotations/        # instances_val2017.json
│   └── sample/
│       └── sample_500.json      # IDs das 500 imagens da amostra estratificada
├── src/
│   ├── __init__.py
│   ├── sampling.py              # Módulo 1: Amostragem estratificada
│   ├── prompts.py               # Módulo 2: Geração de prompts
│   ├── grounding.py             # Módulo 3: Grounding DINO inference
│   ├── segmentation.py          # Módulo 4: SAM 2.1 inference
│   ├── matching.py              # Módulo 5: Hungarian matching GT ↔ predição
│   ├── metrics.py               # Módulo 6: Cálculo de métricas (3 camadas)
│   ├── pipeline.py              # Orquestrador: encadeia módulos 1→6
│   └── visualization.py         # Módulo 7: Visualização de falhas (E5)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_run_experiments.ipynb
│   └── 03_analysis.ipynb
├── results/
│   ├── raw/                     # Resultados brutos (JSON por imagem)
│   ├── tables/                  # Tabelas agregadas (CSV)
│   └── figures/                 # Gráficos e exemplos visuais
├── tests/
│   ├── test_matching.py
│   ├── test_metrics.py
│   └── test_prompts.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 2. Dependências e Versões

### requirements.txt

```
# Core
torch>=2.1.0
torchvision>=0.16.0

# Grounding DINO
groundingdino-py>=0.4.0        # ou clone do repo oficial IDEA-Research/GroundingDINO

# SAM 2
sam2>=0.4.0                     # Meta: facebookresearch/sam2

# COCO
pycocotools>=2.0.7
fiftyone>=0.23.0                # Opcional: visualização e exploração de dados

# Métricas e matching
scipy>=1.11.0                   # linear_sum_assignment (Hungarian)
scikit-image>=0.21.0            # Boundary F1 (find_contours)
numpy>=1.24.0

# Análise e visualização
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuração
pyyaml>=6.0
tqdm>=4.65.0

# Testes
pytest>=7.4.0
```

### Pesos dos modelos (download)

| Modelo | Arquivo | Fonte | Tamanho |
|--------|---------|-------|---------|
| Grounding DINO (Swin-B) | `groundingdino_swinb_cogcoor.pth` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | ~938 MB |
| SAM 2.1 (ViT-H) | `sam2.1_hiera_large.pt` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | ~2.4 GB |

---

## 3. Especificação dos Módulos

### Módulo 1 — `sampling.py` — Amostragem Estratificada

**Entrada**: `instances_val2017.json`  
**Saída**: `sample_500.json` (lista de image_ids)

```python
# Pseudocódigo
TARGET_CLASSES = [1, 18, 17, 3, 2, 62, 44, 47, 53, 59]  # COCO category IDs
# person=1, dog=18, cat=17, car=3, bicycle=2, chair=62, bottle=44, cup=47, apple=53, pizza=59
MIN_INSTANCES_PER_CLASS = 30
SAMPLE_SIZE = 500

def stratified_sample(annotations_path, target_classes, min_instances, sample_size):
    """
    1. Carregar annotations COCO
    2. Para cada classe-alvo, listar todas as imagens que contêm ≥1 instância
    3. Garantir que o conjunto final tenha ≥min_instances instâncias por classe
    4. Completar até sample_size com imagens adicionais (priorizando cobertura multi-classe)
    5. Salvar lista de image_ids
    6. Gerar relatório de cobertura: instâncias por classe na amostra
    """
```

**Validação**: Após gerar a amostra, imprimir tabela de cobertura:

| Classe | Instâncias no val2017 completo | Instâncias na amostra | % |
|--------|-------------------------------|-----------------------|---|
| person | ~11.000 | ≥30 | — |
| ... | ... | ... | — |

---

### Módulo 2 — `prompts.py` — Geração de Prompts

**Entrada**: Lista de classes  
**Saída**: Dicionário `{class_name: {lang: {prompt_type: prompt_text}}}`

```python
PROMPT_TEMPLATES = {
    "simple":      {"en": "{class}",                    "pt": "{class_pt}"},
    "direct":      {"en": "segment the {class}",        "pt": "segmentar o {class_pt}"},
    "contextual":  {"en": "the {class} in the image",   "pt": "o {class_pt} na imagem"},
    "object":      {"en": "object: {class}",            "pt": "objeto: {class_pt}"},
}

CLASS_MAP = {
    "person":  "pessoa",
    "dog":     "cachorro",
    "cat":     "gato",
    "car":     "carro",
    "bicycle": "bicicleta",
    "chair":   "cadeira",
    "bottle":  "garrafa",
    "cup":     "copo",
    "apple":   "maçã",
    "pizza":   "pizza",
}

def generate_all_prompts() -> dict:
    """
    Retorna 10 classes × 4 tipos × 2 idiomas = 80 prompts
    """
```

**Nota sobre gênero**: Alguns nomes em PT são femininos ("pessoa", "bicicleta", "cadeira", "garrafa", "maçã"). O template `"segmentar o {class_pt}"` deve ser `"segmentar a {class_pt}"` para femininos. Implementar mapeamento de artigos:

```python
ARTICLE_MAP = {
    "pessoa": "a", "cachorro": "o", "gato": "o", "carro": "o",
    "bicicleta": "a", "cadeira": "a", "garrafa": "a", "copo": "o",
    "maçã": "a", "pizza": "a",
}
```

Resultado corrigido: `"segmentar a cadeira"`, `"a cadeira na imagem"`, etc.

---

### Módulo 3 — `grounding.py` — Grounding DINO Inference

**Entrada**: Imagem, prompt textual  
**Saída**: Lista de `{box: [x1,y1,x2,y2], confidence: float, prompt: str}`

```python
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

def load_grounding_dino(weights_path: str, device: str = "cuda") -> Model:
    """Carrega GDINO Swin-B com pesos pré-treinados."""

def detect(model, image: np.ndarray, prompt: str) -> list[dict]:
    """
    1. Pre-processar imagem
    2. Executar inferência com prompt
    3. Filtrar boxes: confidence >= BOX_THRESHOLD e text_score >= TEXT_THRESHOLD
    4. Retornar lista de detecções
    """
```

**Formato de saída por imagem** (salvo em `results/raw/`):

```json
{
    "image_id": 139,
    "prompt": "dog",
    "language": "en",
    "prompt_type": "simple",
    "detections": [
        {"box": [120.5, 45.2, 380.1, 290.7], "confidence": 0.87},
        {"box": [450.0, 100.3, 600.2, 350.8], "confidence": 0.72}
    ]
}
```

---

### Módulo 4 — `segmentation.py` — SAM 2.1 Inference

**Entrada**: Imagem, lista de bounding boxes  
**Saída**: Lista de máscaras binárias (numpy arrays)

```python
def load_sam2(weights_path: str, device: str = "cuda") -> SAM2ImagePredictor:
    """Carrega SAM 2.1 ViT-H."""

def segment_boxes(predictor, image: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
    """
    Para cada box:
    1. Converter box para formato SAM (xyxy)
    2. predictor.predict(box=box_tensor)
    3. Selecionar máscara com maior predicted_iou (SAM retorna 3 candidatas)
    4. Retornar máscara binária (H, W)
    """
```

**Decisão de design**: SAM 2.1 retorna 3 máscaras candidatas por box com scores diferentes. **Usar a máscara com maior `predicted_iou`** — é o comportamento padrão e mais conservador.

---

### Módulo 5 — `matching.py` — Hungarian Matching

**Entrada**: Máscaras geradas, máscaras GT (mesma classe, mesma imagem)  
**Saída**: Pares matched, falsos positivos, misses

```python
from scipy.optimize import linear_sum_assignment

IOU_MATCH_THRESHOLD = 0.10  # IoU mínimo para aceitar um match

def compute_mask_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """IoU entre duas máscaras binárias."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / union if union > 0 else 0.0

def hungarian_match(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> dict:
    """
    1. Calcular matriz de IoU pairwise (N_pred × N_gt)
    2. Matriz de custo = 1 - IoU (minimizar custo = maximizar IoU)
    3. Resolver assignment com linear_sum_assignment
    4. Filtrar matches com IoU < IOU_MATCH_THRESHOLD → reclassificar como FP
    5. Retornar:
       - matched: [(pred_idx, gt_idx, iou), ...]
       - false_positives: [pred_idx, ...]  (sem GT correspondente)
       - misses: [gt_idx, ...]  (GT sem predição correspondente)
    """
```

---

### Módulo 6 — `metrics.py` — Cálculo de Métricas

**Entrada**: Resultados do matching por imagem  
**Saída**: Métricas por instância, por classe, por idioma, por tipo de prompt

```python
def compute_box_iou(box_pred, box_gt) -> float:
    """IoU entre duas bounding boxes [x1,y1,x2,y2]."""

def compute_dice(mask_pred, mask_gt) -> float:
    """Dice = 2|P∩G| / (|P|+|G|)"""

def compute_boundary_f1(mask_pred, mask_gt, tolerance=2) -> float:
    """
    1. Extrair contornos de ambas as máscaras (skimage.measure.find_contours)
    2. Para cada pixel de contorno pred, verificar se existe pixel de contorno GT dentro de 'tolerance' pixels
    3. Calcular precision e recall de borda
    4. F1 = 2 * precision * recall / (precision + recall)
    """

def classify_utility(mask_iou: float) -> str:
    """
    'good'       se IoU >= 0.75
    'correctable' se 0.50 <= IoU < 0.75
    'bad'        se IoU < 0.50
    """

def aggregate_results(all_results: list[dict]) -> pd.DataFrame:
    """
    Agregar por:
    - Classe
    - Idioma (en/pt)
    - Tipo de prompt (simple/direct/contextual/object)
    - Combinação idioma × tipo
    Produzir tabelas para cada experimento (E1–E5)
    """
```

---

### Módulo 7 — `pipeline.py` — Orquestrador

```python
def run_experiment(config_path: str):
    """
    1. Carregar configuração (experiment.yaml)
    2. Carregar amostra de imagens (sample_500.json)
    3. Gerar todos os prompts (80 condições)
    4. Carregar modelos (GDINO + SAM 2.1)
    5. Para cada imagem na amostra:
       a. Carregar imagem e anotações GT
       b. Para cada classe presente na imagem:
          c. Para cada condição (idioma × tipo de prompt):
             i.   Executar grounding → boxes
             ii.  Executar segmentação → máscaras
             iii. Executar matching → pares, FP, misses
             iv.  Calcular métricas (3 camadas)
             v.   Salvar resultado bruto (JSON)
    6. Agregar resultados → tabelas CSV
    7. Gerar figuras → results/figures/
    """
```

**Estimativa de execução**:

| Componente | Por condição | Total (500 imgs × 80 cond.) |
|------------|-------------|----------------------------|
| GDINO inference | ~0.15s/img | ~6.000s (~1h40) |
| SAM 2.1 inference | ~0.10s/box | ~4.000s (~1h) estimado |
| Matching + métricas | ~0.01s | desprezível |
| **Total estimado** | | **~3h na A100** |

**Nota**: Nem todas as 80 condições se aplicam a cada imagem. Só classes presentes na imagem são testadas. O total real será menor.

---

## 4. Configuração — `experiment.yaml`

```yaml
experiment:
  name: "vlm-prelabeling-pt-en"
  seed: 42

data:
  coco_root: "data/coco"
  annotations: "data/coco/annotations/instances_val2017.json"
  images_dir: "data/coco/val2017"
  sample_file: "data/sample/sample_500.json"
  sample_size: 500
  min_instances_per_class: 30

classes:
  - {id: 1,  en: "person",   pt: "pessoa",    article: "a"}
  - {id: 18, en: "dog",      pt: "cachorro",  article: "o"}
  - {id: 17, en: "cat",      pt: "gato",      article: "o"}
  - {id: 3,  en: "car",      pt: "carro",     article: "o"}
  - {id: 2,  en: "bicycle",  pt: "bicicleta", article: "a"}
  - {id: 62, en: "chair",    pt: "cadeira",   article: "a"}
  - {id: 44, en: "bottle",   pt: "garrafa",   article: "a"}
  - {id: 47, en: "cup",      pt: "copo",      article: "o"}
  - {id: 53, en: "apple",    pt: "maçã",      article: "a"}
  - {id: 59, en: "pizza",    pt: "pizza",     article: "a"}

prompts:
  types: ["simple", "direct", "contextual", "object"]
  languages: ["en", "pt"]

models:
  grounding_dino:
    weights: "weights/groundingdino_swinb_cogcoor.pth"
    config: "configs/GroundingDINO_SwinB_cfg.py"
    box_threshold: 0.30
    text_threshold: 0.25
  sam2:
    weights: "weights/sam2.1_hiera_large.pt"
    model_cfg: "configs/sam2.1/sam2.1_hiera_l.yaml"

matching:
  algorithm: "hungarian"
  iou_threshold: 0.10     # Mínimo para aceitar match

metrics:
  utility_thresholds:
    good: 0.75
    correctable: 0.50
  boundary_f1_tolerance: 2  # pixels
  conditional_analysis:
    - 0.50   # COCO standard
    - 0.75   # COCO strict

output:
  raw_dir: "results/raw"
  tables_dir: "results/tables"
  figures_dir: "results/figures"
```

---

## 5. Saídas Esperadas por Experimento

### E1 — Qualidade geral

| Arquivo | Formato | Conteúdo |
|---------|---------|----------|
| `e1_overall.csv` | CSV | mIoU, Dice, GSR, Boundary F1 médios por classe (prompt simples EN como baseline) |
| `e1_utility_distribution.csv` | CSV | % boa/corrigível/ruim por classe |
| `e1_class_comparison.png` | PNG | Gráfico de barras: mIoU por classe |

### E2 — Inglês vs Português

| Arquivo | Conteúdo |
|---------|----------|
| `e2_lang_comparison.csv` | mIoU, Dice por classe × idioma |
| `e2_lang_delta.csv` | Δ(EN-PT) por classe com intervalo de confiança |
| `e2_heatmap.png` | Heatmap: classe × idioma × mIoU |

### E3 — Formulação do prompt

| Arquivo | Conteúdo |
|---------|----------|
| `e3_prompt_type.csv` | mIoU por tipo × idioma |
| `e3_interaction.png` | Gráfico: tipo × idioma (buscar interação) |

### E4 — Diagnóstico grounding vs segmentação

| Arquivo | Conteúdo |
|---------|----------|
| `e4_conditional.csv` | Mask IoU condicionado a Box IoU ≥ 0.50 e ≥ 0.75 |
| `e4_scatter.png` | Scatter plot: Box IoU (x) × Mask IoU (y), colorido por idioma |
| `e4_error_source.csv` | % de erros atribuídos a grounding vs segmentação |

### E5 — Análise qualitativa

| Arquivo | Conteúdo |
|---------|----------|
| `e5_failure_examples/` | 5-8 imagens com GT, box, máscara gerada e rótulo de tipo de falha |
| `e5_failure_taxonomy.csv` | Contagem por tipo de falha (grounding miss, box incompleta, máscara excessiva, máscara incompleta, ambiguidade textual, erro por idioma) |

---

## 6. Scripts de Setup

### Download COCO val2017

```bash
mkdir -p data/coco
cd data/coco

# Imagens
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Anotações
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Download pesos dos modelos

```bash
mkdir -p weights

# Grounding DINO Swin-B
wget -O weights/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# SAM 2.1 ViT-H (Hiera Large)
wget -O weights/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Instalação

```bash
# Criar environment
python -m venv .venv
source .venv/bin/activate

# Instalar PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar Grounding DINO
pip install groundingdino-py

# Instalar SAM 2
pip install sam2

# Instalar dependências restantes
pip install -r requirements.txt
```

---

## 7. Validações Pré-Experimento

Antes de executar o pipeline completo, rodar os seguintes checks:

| # | Check | Comando | Critério de sucesso |
|---|-------|---------|---------------------|
| 1 | COCO amostra cobre todas as classes | `python -m src.sampling --validate` | ≥30 instâncias/classe |
| 2 | GDINO carrega e faz inferência | `python -c "from src.grounding import load_grounding_dino; ..."` | Sem erro CUDA |
| 3 | SAM 2.1 carrega e segmenta uma box | `python -c "from src.segmentation import load_sam2; ..."` | Máscara com shape correto |
| 4 | Prompts PT geram artigos corretos | `python -m pytest tests/test_prompts.py` | "segmentar a cadeira" (não "o cadeira") |
| 5 | Hungarian matching funciona | `python -m pytest tests/test_matching.py` | Testes com 0, 1, N pred × M GT |
| 6 | Métricas calculam corretamente | `python -m pytest tests/test_metrics.py` | IoU(mask, mask) == 1.0; Dice(∅, ∅) == 0.0 |
| 7 | GPU tem memória suficiente | `nvidia-smi` | ≥30 GB livres para GDINO + SAM simultâneos |

---

## 8. Ordem de Implementação Sugerida

| Fase | Módulos | Estimativa | Dependência |
|------|---------|------------|-------------|
| 1 | `sampling.py` + `prompts.py` | Setup rápido | Nenhuma |
| 2 | `grounding.py` (carregar modelo, inferir 1 imagem) | Testar GDINO | Download pesos |
| 3 | `segmentation.py` (carregar SAM, segmentar 1 box) | Testar SAM | Download pesos |
| 4 | `matching.py` + `metrics.py` + testes | Lógica pura | Nenhuma (pode ser paralelo a 2-3) |
| 5 | `pipeline.py` (integrar 1→4, rodar em 10 imagens) | Smoke test | Fases 1-4 |
| 6 | Executar pipeline completo (500 imgs × 80 cond.) | ~3h na A100 | Fase 5 |
| 7 | `visualization.py` + análise (notebooks) | Pós-execução | Fase 6 |
