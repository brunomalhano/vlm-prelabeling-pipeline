"""VLM Pre-Labeling Pipeline — Results Dashboard.

Usage:
    streamlit run dashboard.py -- --results-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.basedatatypes as _pbd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Plotly 6+ encodes numeric arrays as base64 binary (bdata) which older
# Streamlit/Plotly.js frontends cannot decode, resulting in blank charts.
# Disabling the conversion keeps data as plain JSON arrays.
_pbd.convert_to_base64 = lambda obj: None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_all_raw(raw_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for f in sorted(raw_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data.get("results", []):
            r["image_id"] = data["image_id"]
            records.append(r)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Consistent palette and formatting
# ---------------------------------------------------------------------------

METRIC_COLORS = {"Mask IoU": "#636EFA", "Dice": "#EF553B", "Boundary F1": "#00CC96"}
LANG_COLORS = {"en": "#636EFA", "pt": "#EF553B"}
UTIL_COLORS = {"good": "#2ECC71", "correctable": "#F39C12", "bad": "#E74C3C"}
UTIL_LABELS = {"good": "Bom (IoU ≥ 0.75)", "correctable": "Corrigível (0.50–0.75)", "bad": "Ruim (< 0.50)"}

CHART_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=20, r=20, t=40, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, **kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VLM Pipeline — Resultados",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

args = _parse_args()
RESULTS = Path(args.results_dir)
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
RAW = RESULTS / "raw"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔬 VLM Pipeline")
st.sidebar.markdown("**Pré-Rotulagem com Grounding DINO + SAM 2.1**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navegação",
    [
        "Visão Geral",
        "E1 — Baseline",
        "E3 — Tipo de Prompt",
        "E4 — Grounding vs Segmentação",
        "E5 — Taxonomia de Falhas",
        "Explorador por Imagem",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("Pipeline executado em 21/mai/2026")
st.sidebar.caption("500 imagens · 10 classes · 4 tipos de prompt · EN")

# ---------------------------------------------------------------------------
# Helper: metric card row
# ---------------------------------------------------------------------------

def metric_cards(cols_spec: list[tuple[str, str, str | None]]):
    """Render a row of st.metric cards. Each tuple: (label, value, delta)."""
    cols = st.columns(len(cols_spec))
    for col, (label, value, delta) in zip(cols, cols_spec):
        col.metric(label, value, delta)


# ===================================================================
# PAGE: Visão Geral
# ===================================================================

if page == "Visão Geral":
    st.title("Resultados do Pipeline de Pré-Rotulagem VLM")
    st.markdown(
        "Avaliação de prompts EN/PT para pré-rotulagem automática com "
        "**Grounding DINO (Swin-B)** + **SAM 2.1 (Hiera Large)** sobre COCO val2017."
    )

    # Top-level metrics
    e1 = load_csv(TABLES / "e1_overall.csv")
    e5 = load_csv(TABLES / "e5_failure_taxonomy.csv")
    total_instances = int(e1["count"].sum())
    avg_miou = e1["mean_mask_iou"].mean()
    avg_det = e1["detection_rate"].mean()
    grounding_miss_pct = float(e5.loc[e5["failure_type"] == "grounding_miss", "percentage"].values[0])

    e3_best_miou = None
    e3_path = TABLES / "e3_prompt_type.csv"
    if e3_path.exists():
        _e3 = load_csv(e3_path)
        e3_best_miou = float(_e3["mean_mask_iou"].max())

    metric_cards([
        ("Imagens Processadas", "500", None),
        ("Total de Instâncias", f"{total_instances:,}", None),
        ("mIoU Baseline (simple EN)", f"{avg_miou:.3f}", None),
        ("Taxa de Detecção Média", f"{avg_det:.1%}", None),
        ("Grounding Misses", f"{grounding_miss_pct}%", "gargalo principal"),
    ])

    st.divider()

    # ── Lollipop: mIoU per class (cleaner than grouped bars) ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("mIoU por Categoria (Baseline EN)")
        e1_sorted = e1.sort_values("mean_mask_iou", ascending=True)
        fig = go.Figure()
        # stem lines
        for _, row in e1_sorted.iterrows():
            fig.add_shape(
                type="line", x0=0, x1=row["mean_mask_iou"],
                y0=row["class_name"], y1=row["class_name"],
                line=dict(color="rgba(99,110,250,0.4)", width=2),
            )
        # dots
        fig.add_trace(go.Scatter(
            x=e1_sorted["mean_mask_iou"], y=e1_sorted["class_name"],
            mode="markers+text",
            marker=dict(size=12, color="#636EFA"),
            text=e1_sorted["mean_mask_iou"].apply(lambda v: f"{v:.3f}"),
            textposition="middle right",
            showlegend=False,
        ))
        # reference lines
        fig.add_vline(x=0.75, line_dash="dot", line_color="#2ECC71",
                      annotation_text="Bom", annotation_position="top")
        fig.add_vline(x=0.50, line_dash="dot", line_color="#F39C12",
                      annotation_text="Corrigível", annotation_position="top")
        apply_layout(fig, height=400, xaxis_title="mIoU", yaxis_title="",
                     xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição de Utilidade")
        util = load_csv(TABLES / "e1_utility_distribution.csv")
        util_sorted = util.sort_values("good", ascending=True)
        fig2 = go.Figure()
        for col_name, label, color in [
            ("bad", UTIL_LABELS["bad"], UTIL_COLORS["bad"]),
            ("correctable", UTIL_LABELS["correctable"], UTIL_COLORS["correctable"]),
            ("good", UTIL_LABELS["good"], UTIL_COLORS["good"]),
        ]:
            fig2.add_trace(go.Bar(
                y=util_sorted["class_name"], x=util_sorted[col_name],
                name=label, orientation="h", marker_color=color,
                text=util_sorted[col_name].apply(lambda v: f"{v:.0f}%"),
                textposition="inside",
            ))
        fig2.update_layout(barmode="stack")
        apply_layout(fig2, height=400, xaxis_title="% das instâncias", yaxis_title="",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Radar: multi-metric profile per class ──
    st.subheader("Perfil Multi-Métrica por Categoria")
    radar_classes = st.multiselect(
        "Selecione categorias para comparar",
        e1["class_name"].tolist(),
        default=["cat", "dog", "person", "apple", "chair"],
    )
    if radar_classes:
        metrics = ["mean_mask_iou", "mean_dice", "mean_boundary_f1", "detection_rate"]
        metric_labels = ["Mask IoU", "Dice", "Boundary F1", "Taxa Detecção"]
        fig_radar = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, cls in enumerate(radar_classes):
            row = e1[e1["class_name"] == cls].iloc[0]
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=metric_labels + [metric_labels[0]],
                fill="toself", name=cls, opacity=0.6,
                line=dict(color=colors[i % len(colors)]),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
        )
        apply_layout(fig_radar, showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Quick summary cards
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Resumo: Tipos de Prompt (E3)")
        if e3_path.exists():
            _e3 = load_csv(e3_path)
            _best = _e3.loc[_e3["mean_mask_iou"].idxmax()]
            _worst = _e3.loc[_e3["mean_mask_iou"].idxmin()]
            _ratio = _best["mean_mask_iou"] / _worst["mean_mask_iou"]
            st.metric("Melhor Prompt", _best["prompt_type"],
                      f"mIoU {_best['mean_mask_iou']:.3f}")
            st.metric("Pior Prompt", _worst["prompt_type"],
                      f"mIoU {_worst['mean_mask_iou']:.3f}")
            st.metric("Razão Melhor/Pior", f"{_ratio:.1f}×")
        else:
            st.info("Dados E3 ainda não disponíveis.")

    with col4:
        st.subheader("Resumo: Gargalo (E4)")
        _e4_path = TABLES / "e4_error_source.csv"
        if _e4_path.exists():
            e4_err = load_csv(_e4_path)
            ge = float(e4_err["grounding_error_pct"].values[0])
            se = float(e4_err["segmentation_error_pct"].values[0])
            st.metric("Erros de Grounding", f"{ge}%", "gargalo")
            st.metric("Erros de Segmentação", f"{se}%", "baixo")
        else:
            st.info("Dados E4 ainda não disponíveis.")


# ===================================================================
# PAGE: E1 — Baseline
# ===================================================================

elif page == "E1 — Baseline":
    st.title("E1 — Qualidade Baseline (Prompts Simples em EN)")
    st.markdown("Baseline usando o tipo de prompt mais simples (`simple`) em inglês nas 500 imagens.")

    e1 = load_csv(TABLES / "e1_overall.csv")

    # Interactive table
    st.subheader("Métricas por Categoria")
    display_df = e1.rename(columns={
        "class_name": "Categoria",
        "mean_mask_iou": "mIoU",
        "mean_dice": "Dice",
        "mean_boundary_f1": "BF1",
        "detection_rate": "Taxa Detecção",
        "count": "Instâncias",
    })
    display_df = display_df.sort_values("mIoU", ascending=False)
    st.dataframe(
        display_df.style.format({
            "mIoU": "{:.3f}",
            "Dice": "{:.3f}",
            "BF1": "{:.3f}",
            "Taxa Detecção": "{:.1%}",
        }).background_gradient(subset=["mIoU"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Dot plot: 3 metrics side-by-side per class ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Comparação de Métricas")
        e1_sorted = e1.sort_values("mean_mask_iou", ascending=True)
        fig = go.Figure()
        for metric, name, color, symbol in [
            ("mean_mask_iou", "Mask IoU", "#636EFA", "circle"),
            ("mean_dice", "Dice", "#EF553B", "diamond"),
            ("mean_boundary_f1", "Boundary F1", "#00CC96", "square"),
        ]:
            fig.add_trace(go.Scatter(
                x=e1_sorted[metric], y=e1_sorted["class_name"],
                mode="markers", name=name,
                marker=dict(size=11, color=color, symbol=symbol),
            ))
        fig.add_vline(x=0.50, line_dash="dot", line_color="#F39C12", annotation_text="0.50")
        apply_layout(fig, height=420, xaxis_title="Score", yaxis_title="",
                     xaxis_range=[0, 1],
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Taxa de Detecção vs mIoU")
        fig2 = px.scatter(
            e1, x="detection_rate", y="mean_mask_iou",
            size="count", color="class_name",
            labels={
                "detection_rate": "Taxa de Detecção",
                "mean_mask_iou": "mIoU Média",
                "count": "Instâncias",
                "class_name": "Categoria",
            },
            height=420,
        )
        fig2.update_traces(textposition="top center")
        # add quadrant annotations
        fig2.add_hline(y=0.50, line_dash="dot", line_color="rgba(128,128,128,0.5)")
        fig2.add_vline(x=0.70, line_dash="dot", line_color="rgba(128,128,128,0.5)")
        fig2.add_annotation(x=0.85, y=0.85, text="Alta detecção<br>Alta qualidade",
                            showarrow=False, font=dict(size=10, color="green"), opacity=0.5)
        fig2.add_annotation(x=0.45, y=0.20, text="Baixa detecção<br>Baixa qualidade",
                            showarrow=False, font=dict(size=10, color="red"), opacity=0.5)
        apply_layout(fig2, xaxis_tickformat=".0%", xaxis_range=[0.3, 1], yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Utility: percentage bar with value labels ──
    st.subheader("Distribuição de Utilidade por Categoria")
    util = load_csv(TABLES / "e1_utility_distribution.csv")
    util_sorted = util.sort_values("good", ascending=False)
    fig3 = go.Figure()
    for col_name, label, color in [
        ("good", UTIL_LABELS["good"], UTIL_COLORS["good"]),
        ("correctable", UTIL_LABELS["correctable"], UTIL_COLORS["correctable"]),
        ("bad", UTIL_LABELS["bad"], UTIL_COLORS["bad"]),
    ]:
        fig3.add_trace(go.Bar(
            x=util_sorted["class_name"], y=util_sorted[col_name],
            name=label, marker_color=color,
            text=util_sorted[col_name].apply(lambda v: f"{v:.0f}%"),
            textposition="inside",
        ))
    fig3.update_layout(barmode="stack")
    apply_layout(fig3, height=380, xaxis_title="Categoria", yaxis_title="% das instâncias",
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig3, use_container_width=True)

    # Show generated figure
    fig_path = FIGURES / "e1_class_comparison.png"
    if fig_path.exists():
        with st.expander("Figura gerada pelo pipeline"):
            st.image(str(fig_path), use_column_width=True)


# ===================================================================
# PAGE: E2 — Idioma (not rendered — EN-only experiment)
# ===================================================================

# E2 page removed: this run used EN prompts only.
# If a bilingual run is executed in the future, restore this page.
# (E2 visualization code commented out — no e2_delta / e2_comp data available)


# ===================================================================
# PAGE: E3 — Tipo de Prompt
# ===================================================================

elif page == "E3 — Tipo de Prompt":
    st.title("E3 — Comparação de Tipos de Prompt (EN)")
    st.markdown(
        "Impacto do formato de prompt na qualidade de pré-rotulagem. "
        "Quatro tipos avaliados em inglês: **simple**, **direct**, **contextual**, **object-prefix**."
    )

    e3 = load_csv(TABLES / "e3_prompt_type.csv")

    # Table
    st.subheader("Métricas por Tipo de Prompt")
    _rename = {"prompt_type": "Tipo", "mean_mask_iou": "mIoU",
               "mean_dice": "Dice", "mean_boundary_f1": "BF1", "count": "Instâncias"}
    if "language" in e3.columns:
        _rename["language"] = "Idioma"
    e3_display = e3.rename(columns=_rename)
    _fmt_cols = {c: "{:.3f}" for c in ["mIoU", "Dice", "BF1"] if c in e3_display.columns}
    st.dataframe(
        e3_display.style.format(_fmt_cols).background_gradient(subset=["mIoU"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    # ── get EN slice (handles both with and without language column) ──
    e3_en_df = (
        e3[e3["language"] == "en"] if "language" in e3.columns else e3
    ).sort_values("mean_mask_iou", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # ── Lollipop: mIoU per prompt type ──
        st.subheader("mIoU por Tipo de Prompt")
        e3_sorted = e3_en_df.sort_values("mean_mask_iou", ascending=True)
        fig = go.Figure()
        colors_prompt = px.colors.qualitative.Plotly
        for i, (_, row) in enumerate(e3_sorted.iterrows()):
            color = colors_prompt[i % len(colors_prompt)]
            fig.add_shape(
                type="line", x0=0, x1=row["mean_mask_iou"],
                y0=row["prompt_type"], y1=row["prompt_type"],
                line=dict(color=f"rgba({','.join(str(int(c*255)) for c in px.colors.hex_to_rgb(color))},0.4)"
                          if color.startswith("#") else "rgba(99,110,250,0.4)", width=3),
            )
        fig.add_trace(go.Scatter(
            x=e3_sorted["mean_mask_iou"], y=e3_sorted["prompt_type"],
            mode="markers+text",
            marker=dict(size=14, color=colors_prompt[:len(e3_sorted)]),
            text=e3_sorted["mean_mask_iou"].apply(lambda v: f"{v:.3f}"),
            textposition="middle right",
            showlegend=False,
        ))
        apply_layout(fig, height=350, xaxis_title="mIoU", yaxis_title="",
                     xaxis_range=[0, e3_sorted["mean_mask_iou"].max() * 1.25])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ── Grouped bar: all 3 metrics per prompt type ──
        st.subheader("Métricas por Tipo de Prompt")
        fig2 = go.Figure()
        for metric, name, color in [
            ("mean_mask_iou", "Mask IoU", METRIC_COLORS["Mask IoU"]),
            ("mean_dice", "Dice", METRIC_COLORS["Dice"]),
            ("mean_boundary_f1", "Boundary F1", METRIC_COLORS["Boundary F1"]),
        ]:
            if metric in e3_en_df.columns:
                fig2.add_trace(go.Bar(
                    x=e3_en_df["prompt_type"], y=e3_en_df[metric],
                    name=name, marker_color=color,
                    text=e3_en_df[metric].apply(lambda v: f"{v:.3f}"),
                    textposition="outside",
                ))
        fig2.update_layout(barmode="group")
        apply_layout(fig2, height=350, xaxis_title="Tipo de Prompt", yaxis_title="Score",
                     yaxis_range=[0, max(e3_en_df["mean_mask_iou"].max() * 1.3, 0.6)],
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Delta bar: each type vs simple (reference) ──
    st.subheader("Delta vs Prompt Simple")
    simple_miou = e3_en_df.loc[e3_en_df["prompt_type"] == "simple", "mean_mask_iou"]
    if len(simple_miou) > 0:
        ref = float(simple_miou.values[0])
        _delta = e3_en_df.copy()
        _delta["delta"] = _delta["mean_mask_iou"] - ref
        _delta_other = _delta[_delta["prompt_type"] != "simple"].sort_values("delta")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=_delta_other["prompt_type"],
            y=_delta_other["delta"],
            marker_color=[
                "#2ECC71" if v >= 0 else "#E74C3C" for v in _delta_other["delta"]
            ],
            text=_delta_other["delta"].apply(lambda v: f"{v:+.3f}"),
            textposition="outside",
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray",
                       annotation_text="simple (referência)", annotation_position="right")
        apply_layout(fig3, height=300, xaxis_title="Tipo de Prompt",
                     yaxis_title="Δ mIoU vs simple", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    _best_type = e3_en_df.iloc[0]["prompt_type"]
    _worst_type = e3_en_df.iloc[-1]["prompt_type"]
    _ratio = e3_en_df["mean_mask_iou"].max() / e3_en_df["mean_mask_iou"].min()
    st.info(
        f"**Descoberta principal**: O prompt `{_best_type}` é **{_ratio:.1f}× mais eficaz** "
        f"que `{_worst_type}`. Prompts mais elaborados degradam o desempenho do Grounding DINO — "
        "a inferência mais simples da classe supera descrições contextuais."
    )

    fig_path = FIGURES / "e3_prompt_interaction.png"
    if fig_path.exists():
        with st.expander("Figura gerada pelo pipeline"):
            st.image(str(fig_path), use_column_width=True)


# ===================================================================
# PAGE: E4 — Grounding vs Segmentação
# ===================================================================

elif page == "E4 — Grounding vs Segmentação":
    st.title("E4 — Diagnóstico: Grounding vs Segmentação")
    st.markdown("Onde está o gargalo: na detecção do objeto ou na geração da máscara?")

    e4_cond = load_csv(TABLES / "e4_conditional.csv")
    e4_err = load_csv(TABLES / "e4_error_source.csv")

    # Error source metrics
    ge = float(e4_err["grounding_errors"].values[0])
    se = float(e4_err["segmentation_errors"].values[0])
    ge_pct = float(e4_err["grounding_error_pct"].values[0])
    se_pct = float(e4_err["segmentation_error_pct"].values[0])
    total = int(e4_err["total_matched"].values[0])
    correct = total - ge - se

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Correspondências", f"{total:,}")
    col2.metric("Erros de Grounding", f"{int(ge):,}", f"{ge_pct}%")
    col3.metric("Erros de Segmentação", f"{int(se):,}", f"{se_pct}%")

    col_a, col_b = st.columns(2)

    with col_a:
        # ── Horizontal stacked bar (replaces pie) ──
        st.subheader("Fonte de Erros")
        fig = go.Figure()
        categories = ["Correto", "Erro Grounding", "Erro Segmentação"]
        values = [correct, ge, se]
        colors_err = ["#2ECC71", "#E74C3C", "#F39C12"]
        for cat, val, color in zip(categories, values, colors_err):
            pct = val / total * 100
            fig.add_trace(go.Bar(
                y=["Pipeline"], x=[val], name=f"{cat} ({pct:.1f}%)",
                orientation="h", marker_color=color,
                text=f"{int(val):,}", textposition="inside",
                textfont=dict(size=14, color="white"),
            ))
        fig.update_layout(barmode="stack")
        apply_layout(fig, height=180, xaxis_title="Instâncias", yaxis_title="",
                     legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        # ── Waterfall: error decomposition ──
        st.subheader("Decomposição de Erros")
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            x=["Total Matched", "− Grounding Errors", "− Seg Errors", "Corretos"],
            y=[total, -ge, -se, 0],
            measure=["absolute", "relative", "relative", "total"],
            connector=dict(line=dict(color="rgba(128,128,128,0.3)")),
            decreasing=dict(marker_color="#E74C3C"),
            increasing=dict(marker_color="#2ECC71"),
            totals=dict(marker_color="#2ECC71"),
            text=[f"{total:,}", f"−{int(ge):,}", f"−{int(se):,}", f"{int(correct):,}"],
            textposition="outside",
        ))
        apply_layout(fig_wf, height=350, yaxis_title="Instâncias")
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_b:
        st.subheader("Qualidade Condicional da Máscara")
        st.markdown("Quando o grounding é bem-sucedido, qual a qualidade do SAM 2.1?")

        # ── Bullet chart for conditional quality ──
        fig_cond = make_subplots(rows=len(e4_cond), cols=1,
                                 subplot_titles=[f"Box IoU ≥ {row['box_iou_threshold']}"
                                                 for _, row in e4_cond.iterrows()])
        for i, (_, row) in enumerate(e4_cond.iterrows()):
            r = i + 1
            # background ranges
            fig_cond.add_trace(go.Bar(x=[1], y=[""], orientation="h",
                                      marker_color="rgba(231,76,60,0.15)",
                                      showlegend=False, hoverinfo="skip"), row=r, col=1)
            fig_cond.add_trace(go.Bar(x=[0.75], y=[""], orientation="h",
                                      marker_color="rgba(243,156,18,0.15)",
                                      showlegend=False, hoverinfo="skip"), row=r, col=1)
            fig_cond.add_trace(go.Bar(x=[0.50], y=[""], orientation="h",
                                      marker_color="rgba(46,204,113,0.15)",
                                      showlegend=False, hoverinfo="skip"), row=r, col=1)
            # metric markers
            for metric, name, color, symbol in [
                ("mean_mask_iou", "Mask IoU", "#636EFA", "diamond"),
                ("mean_dice", "Dice", "#EF553B", "circle"),
                ("mean_boundary_f1", "BF1", "#00CC96", "square"),
            ]:
                fig_cond.add_trace(go.Scatter(
                    x=[row[metric]], y=[""],
                    mode="markers+text", name=name if i == 0 else None,
                    marker=dict(size=14, color=color, symbol=symbol),
                    text=[f"{row[metric]:.3f}"], textposition="top center",
                    showlegend=(i == 0),
                ), row=r, col=1)
            fig_cond.update_xaxes(range=[0, 1], row=r, col=1)

        apply_layout(fig_cond, height=300,
                     legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5))
        fig_cond.update_layout(barmode="overlay")
        st.plotly_chart(fig_cond, use_container_width=True)

        st.success(
            "Quando o grounding acerta (box IoU ≥ 0.50), o SAM 2.1 produz máscaras "
            "com **mIoU 0.84** e **Dice 0.90** — qualidade excelente. "
            "O gargalo está no grounding, não na segmentação."
        )

    # ── Box vs Mask scatter with marginal histograms ──
    st.subheader("Box IoU vs Mask IoU (dados brutos)")
    with st.spinner("Carregando dados brutos..."):
        raw_df = load_all_raw(RAW)
    if raw_df.empty or "detected" not in raw_df.columns:
        st.info("Dados brutos (raw JSON) não disponíveis nesta execução — scatter plot omitido.")
    elif "box_iou" in raw_df.columns and "mask_iou" in raw_df.columns:
        detected = raw_df[raw_df["detected"] == True].copy()  # noqa: E712
        sample = detected.sample(min(5000, len(detected)), random_state=42)
        fig3 = px.scatter(
            sample, x="box_iou", y="mask_iou",
            color="class_name", opacity=0.35,
            marginal_x="histogram", marginal_y="histogram",
            labels={"box_iou": "Box IoU", "mask_iou": "Mask IoU", "class_name": "Categoria"},
            height=550,
        )
        fig3.add_hline(y=0.50, line_dash="dot", line_color="rgba(243,156,18,0.6)",
                       annotation_text="Mask IoU = 0.50")
        fig3.add_vline(x=0.50, line_dash="dot", line_color="rgba(243,156,18,0.6)",
                       annotation_text="Box IoU = 0.50")
        # add diagonal reference
        fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(color="rgba(128,128,128,0.3)", dash="dash"))
        apply_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    fig_path = FIGURES / "e4_box_vs_mask.png"
    if fig_path.exists():
        with st.expander("Figura gerada pelo pipeline"):
            st.image(str(fig_path), use_column_width=True)


# ===================================================================
# PAGE: E5 — Taxonomia de Falhas
# ===================================================================

elif page == "E5 — Taxonomia de Falhas":
    st.title("E5 — Taxonomia de Falhas")
    st.markdown("Distribuição dos modos de falha entre todas as instâncias que não atingiram qualidade 'Bom'.")

    e5 = load_csv(TABLES / "e5_failure_taxonomy.csv")

    col1, col2 = st.columns(2)

    with col1:
        # ── Treemap (replaces pie) ──
        st.subheader("Distribuição de Falhas")
        e5_display = e5.copy()
        e5_display["label"] = e5_display.apply(
            lambda r: f"{r['failure_type']}<br>{r['count']:,} ({r['percentage']}%)", axis=1)
        failure_colors = {
            "grounding_miss": "#E74C3C",
            "false_positive": "#E67E22",
            "mask_incomplete": "#F39C12",
            "box_incomplete": "#3498DB",
            "mask_excessive": "#9B59B6",
        }
        e5_display["color"] = e5_display["failure_type"].map(failure_colors)
        fig = go.Figure(go.Treemap(
            labels=e5_display["label"],
            parents=[""] * len(e5_display),
            values=e5_display["count"],
            marker=dict(colors=e5_display["color"]),
            textinfo="label",
            textfont=dict(size=14),
        ))
        apply_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ── Horizontal bar with percentage labels ──
        st.subheader("Contagem por Tipo")
        e5_sorted = e5.sort_values("count", ascending=True)
        fig2 = go.Figure()
        bar_colors = [failure_colors.get(ft, "#95A5A6") for ft in e5_sorted["failure_type"]]
        fig2.add_trace(go.Bar(
            y=e5_sorted["failure_type"], x=e5_sorted["count"],
            orientation="h", marker_color=bar_colors,
            text=e5_sorted.apply(lambda r: f"{r['count']:,} ({r['percentage']}%)", axis=1),
            textposition="outside",
        ))
        apply_layout(fig2, height=400, xaxis_title="Contagem", yaxis_title="",
                     showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Pareto: cumulative % ──
    st.subheader("Análise de Pareto")
    e5_pareto = e5.sort_values("count", ascending=False).copy()
    e5_pareto["cumulative_pct"] = e5_pareto["percentage"].cumsum()
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(
        x=e5_pareto["failure_type"], y=e5_pareto["count"],
        marker_color=[failure_colors.get(ft, "#95A5A6") for ft in e5_pareto["failure_type"]],
        name="Contagem",
        text=e5_pareto["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
    ), secondary_y=False)
    fig3.add_trace(go.Scatter(
        x=e5_pareto["failure_type"], y=e5_pareto["cumulative_pct"],
        mode="lines+markers+text", name="% Acumulado",
        line=dict(color="#2C3E50", width=2.5),
        marker=dict(size=8, color="#2C3E50"),
        text=e5_pareto["cumulative_pct"].apply(lambda v: f"{v:.0f}%"),
        textposition="top center",
    ), secondary_y=True)
    fig3.add_hline(y=80, line_dash="dot", line_color="rgba(231,76,60,0.5)",
                   annotation_text="80%", secondary_y=True)
    fig3.update_yaxes(title_text="Contagem", secondary_y=False)
    fig3.update_yaxes(title_text="% Acumulado", range=[0, 105], secondary_y=True)
    apply_layout(fig3, height=380, xaxis_title="Tipo de Falha",
                 legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    st.plotly_chart(fig3, use_container_width=True)

    # Descriptions
    st.subheader("Descrição dos Tipos de Falha")
    descriptions = {
        "grounding_miss": "Objeto presente no ground truth mas **não detectado** pelo Grounding DINO.",
        "false_positive": "Objeto **predito** pelo modelo mas sem correspondente no ground truth.",
        "mask_incomplete": "Bounding box correto mas máscara **subdimensionada** (cobre menos que o objeto).",
        "box_incomplete": "Bounding box detectado mas **pequeno ou deslocado** demais.",
        "mask_excessive": "Bounding box correto mas máscara **superdimensionada** (extravasa o objeto).",
    }
    for _, row in e5.iterrows():
        ft = row["failure_type"]
        st.markdown(f"- **{ft}** ({row['count']:,} — {row['percentage']}%): {descriptions.get(ft, '')}")

    # Failure examples
    st.subheader("Exemplos de Falhas")
    failures_dir = FIGURES / "e5_failures"
    if failures_dir.exists():
        failure_imgs = sorted(failures_dir.glob("failure_*.png"))
        if failure_imgs:
            cols = st.columns(4)
            for i, img in enumerate(failure_imgs):
                cols[i % 4].image(str(img), caption=img.stem, use_column_width=True)

    fig_path = FIGURES / "e5_failure_taxonomy.png"
    if fig_path.exists():
        with st.expander("Figura gerada pelo pipeline"):
            st.image(str(fig_path), use_column_width=True)


# ===================================================================
# PAGE: Explorador por Imagem
# ===================================================================

elif page == "Explorador por Imagem":
    st.title("Explorador por Imagem")
    st.markdown("Explore os resultados de instâncias individuais nos dados brutos.")

    with st.spinner("Carregando todos os dados brutos (44k+ registros)..."):
        raw_df = load_all_raw(RAW)

    if raw_df.empty:
        st.warning("Dados brutos (raw JSON) não disponíveis nesta execução. "
                   "Re-execute o pipeline com upload para blob para gerar os arquivos em `results/raw/`.")
        st.stop()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        classes = sorted(raw_df["class_name"].unique())
        sel_class = st.multiselect("Categoria", classes, default=classes)

    with col_f2:
        ptypes = sorted(raw_df["prompt_type"].unique())
        sel_ptype = st.multiselect("Tipo de Prompt", ptypes, default=ptypes)

    with col_f3:
        utilities = sorted(raw_df["utility_class"].dropna().unique())
        sel_util = st.multiselect("Utilidade", utilities, default=utilities)

    filtered = raw_df[
        (raw_df["class_name"].isin(sel_class))
        & (raw_df["prompt_type"].isin(sel_ptype))
    ]
    if sel_util:
        filtered = filtered[filtered["utility_class"].isin(sel_util)]

    st.markdown(f"**{len(filtered):,}** resultados filtrados de **{len(raw_df):,}** totais")

    # Aggregate stats
    if len(filtered) > 0:
        detected_df = filtered[filtered["detected"] == True]  # noqa: E712
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Detectados", f"{len(detected_df):,} / {len(filtered):,}")
        if len(detected_df) > 0:
            col_s2.metric("mIoU Média", f"{detected_df['mask_iou'].mean():.3f}")
            col_s3.metric("Dice Médio", f"{detected_df['dice'].mean():.3f}")
            col_s4.metric("BF1 Médio", f"{detected_df['boundary_f1'].mean():.3f}")

        st.divider()

        # Distribution of mask_iou
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("Distribuição de Mask IoU")
            if len(detected_df) > 0:
                fig = px.histogram(
                    detected_df,
                    x="mask_iou",
                    nbins=50,
                    color="class_name",
                    labels={"mask_iou": "Mask IoU", "class_name": "Categoria"},
                    height=350,
                )
                fig.add_vline(x=0.75, line_dash="dash", line_color="green", annotation_text="Bom")
                fig.add_vline(x=0.50, line_dash="dash", line_color="orange", annotation_text="Corrigível")
                st.plotly_chart(fig, use_container_width=True)

        with col_d2:
            st.subheader("Contagem por Utilidade")
            if "utility_class" in filtered.columns:
                util_counts = filtered["utility_class"].value_counts().reset_index()
                util_counts.columns = ["utility_class", "count"]
                fig2 = px.bar(
                    util_counts,
                    x="utility_class",
                    y="count",
                    color="utility_class",
                    color_discrete_map={"good": "#2ECC71", "correctable": "#F39C12", "bad": "#E74C3C"},
                    labels={"utility_class": "Utilidade", "count": "Contagem"},
                    height=350,
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        # Raw data table
        st.subheader("Dados Brutos")
        display_cols = [
            "image_id", "class_name", "prompt_type", "prompt_text",
            "detected", "mask_iou", "dice", "boundary_f1", "box_iou", "utility_class",
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[available_cols].sort_values(
                "mask_iou", ascending=False, na_position="last"
            ).head(500).style.format({
                "mask_iou": "{:.3f}",
                "dice": "{:.3f}",
                "boundary_f1": "{:.3f}",
                "box_iou": "{:.3f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
