"""
DataMind AI - Chart Builder & Visualization Tools

Creates publication-quality interactive charts using Plotly.
Implements executive Power BI styling: clean labels, intelligent chart selection
based on data composition & distribution, and formatted human-readable labels.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from src.tools.data_loader import get_dataset

logger = logging.getLogger(__name__)

# Power BI Executive Theme Palette
DATAMIND_PALETTE = [
    "#3b82f6",  # Electric Cobalt (Power BI primary)
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#8b5cf6",  # Modern Purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Rose Pink
    "#6366f1",  # Indigo
    "#14b8a6",  # Teal
]

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17, 17, 24, 0.45)",
    font=dict(color="#f0f0f5", family="'Segoe UI', Inter, -apple-system, sans-serif"),
    title_font=dict(size=14, color="#ffffff", family="'Segoe UI', Inter, sans-serif"),
    margin=dict(l=45, r=30, t=55, b=45),
    hoverlabel=dict(
        bgcolor="#1a1a24",
        bordercolor="#3b82f6",
        font=dict(family="'Segoe UI', Inter, sans-serif", size=12, color="#ffffff"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color="#9898b0"),
    ),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.08)",
        zerolinecolor="rgba(255, 255, 255, 0.08)",
        tickfont=dict(size=11, color="#9898b0"),
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.08)",
        zerolinecolor="rgba(255, 255, 255, 0.08)",
        tickfont=dict(size=11, color="#9898b0"),
    ),
)

# Common business abbreviation dictionary for clean title generation
ABBREVIATION_MAP = {
    "id": "ID",
    "usd": "($)",
    "eur": "(€)",
    "gbp": "(£)",
    "inr": "(₹)",
    "pct": "(%)",
    "perc": "(%)",
    "percentage": "(%)",
    "qty": "Quantity",
    "amt": "Amount",
    "val": "Value",
    "avg": "Average",
    "tot": "Total",
    "cnt": "Count",
    "num": "Number",
    "cat": "Category",
    "desc": "Description",
    "dt": "Date",
    "yr": "Year",
    "mo": "Month",
    "hr": "Hour",
    "min": "Minute",
    "sec": "Second",
    "rev": "Revenue",
    "prod": "Product",
    "cust": "Customer",
    "acc": "Account",
    "tx": "Transaction",
    "txn": "Transaction",
    "org": "Organization",
    "dept": "Department",
    "vol": "Volume",
    "std": "Std Dev",
    "diff": "Difference",
    "prev": "Previous",
    "curr": "Current",
    "freq": "Frequency",
    "src": "Source",
    "dest": "Destination",
    "loc": "Location",
    "pos": "Position",
    "lat": "Latitude",
    "lon": "Longitude",
    "lng": "Longitude",
    "temp": "Temperature",
    "conf": "Confidence",
    "prob": "Probability",
}


def format_label(col: str) -> str:
    """
    Transforms raw dataset column names into polished, human-friendly Power BI labels.
    Examples:
      - 'customer_lifetime_val_usd' -> 'Customer Lifetime Value ($)'
      - 'prod_cat_id' -> 'Product Category ID'
      - 'churn_risk_pct' -> 'Churn Risk (%)'
      - 'monthly_recurring_rev' -> 'Monthly Recurring Revenue'
    """
    if not col:
        return ""
    # Convert camelCase to snake_case first
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', col)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    tokens = re.split(r'[-_.\s]+', s)

    formatted_tokens = []
    for token in tokens:
        if not token:
            continue
        low = token.lower()
        if low in ABBREVIATION_MAP:
            formatted_tokens.append(ABBREVIATION_MAP[low])
        else:
            formatted_tokens.append(token.capitalize())

    result = " ".join(formatted_tokens)
    result = re.sub(r'\s+\(([^\)]+)\)', r' (\1)', result)
    return result.strip() or col


def _is_id_column(col_name: str, series: pd.Series) -> bool:
    """Check if column is an identifier/key rather than an analytical feature."""
    col_lower = col_name.lower()
    if re.search(r'(^id$|_id$|^id_|uuid|guid|hash|token|index|key$|row_num)', col_lower):
        return True
    if len(series) > 10 and series.nunique() == len(series) and series.dtype == 'object':
        return True
    return False


def _clean_chart_spec(obj: Any) -> Any:
    """Recursively convert numpy arrays, bdata dicts, and NaN values to native JSON types."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            try:
                dt_map = {
                    "f8": np.float64, "f4": np.float32,
                    "i8": np.int64, "i4": np.int32,
                    "u8": np.uint64, "u4": np.uint32,
                }
                dtype = dt_map.get(obj["dtype"], np.float64)
                raw = base64.b64decode(obj["bdata"])
                arr = np.frombuffer(raw, dtype=dtype)
                if "shape" in obj and isinstance(obj["shape"], str):
                    shape = tuple(int(s.strip()) for s in obj["shape"].split(","))
                    arr = arr.reshape(shape)
                return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in arr.tolist()]
            except Exception:
                return []
        return {k: _clean_chart_spec(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_chart_spec(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _serialize_plotly_fig(fig: Any) -> dict[str, Any]:
    """Serialize Plotly figure to clean dict with native lists instead of binary bdata."""
    raw_dict = json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
    return _clean_chart_spec(raw_dict)


def create_bar_chart(
    dataset_id: str,
    x: str,
    y: str,
    title: str | None = None,
    orientation: str = "v",
    top_n: int = 15,
) -> dict[str, Any]:
    """Create interactive bar chart with Power BI-style direct data labels and clean titles."""
    df = get_dataset(dataset_id)
    plot_df = df[[x, y]].dropna()

    clean_x = format_label(x)
    clean_y = format_label(y)

    # Aggregate if multiple rows per category
    if plot_df[x].duplicated().any():
        plot_df = plot_df.groupby(x, as_index=False)[y].mean()

    plot_df = plot_df.sort_values(by=y, ascending=(orientation != "h")).head(top_n)

    display_title = title or f"{clean_y} by {clean_x}"

    fig = px.bar(
        plot_df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        orientation=orientation,
        title=display_title,
        color_discrete_sequence=[DATAMIND_PALETTE[0]],
        labels={x: clean_x, y: clean_y},
    )

    # Direct data labels outside bars
    fig.update_traces(
        texttemplate="%{y:,.2s}" if orientation == "v" else "%{x:,.2s}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        opacity=0.92,
        hovertemplate=(
            f"<b>%{{x}}</b><br>{clean_y}: %{{y:,.2f}}<extra></extra>"
            if orientation == "v"
            else f"<b>%{{y}}</b><br>{clean_y}: %{{x:,.2f}}<extra></extra>"
        ),
    )

    layout = dict(**DARK_LAYOUT)
    fig.update_layout(
        **layout,
        xaxis_title=clean_x if orientation == "v" else clean_y,
        yaxis_title=clean_y if orientation == "v" else clean_x,
    )
    return _serialize_plotly_fig(fig)


def create_line_chart(
    dataset_id: str,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create interactive line chart with Power BI-style gradient area fill and smooth spline."""
    df = get_dataset(dataset_id)
    cols = [x, y] + ([color] if color else [])
    plot_df = df[cols].dropna().sort_values(by=x)

    clean_x = format_label(x)
    clean_y = format_label(y)
    display_title = title or f"{clean_y} Trend over {clean_x}"

    fig = px.line(
        plot_df,
        x=x,
        y=y,
        color=color,
        title=display_title,
        color_discrete_sequence=DATAMIND_PALETTE,
        labels={x: clean_x, y: clean_y, (color or ""): format_label(color or "")},
    )

    fig.update_traces(
        line=dict(shape="spline", width=2.5),
        mode="lines+markers",
        marker=dict(size=5),
        fill="tozeroy" if not color else "none",
        fillcolor="rgba(59, 130, 246, 0.08)",
        hovertemplate=f"<b>{clean_x}:</b> %{{x}}<br><b>{clean_y}:</b> %{{y:,.2f}}<extra></extra>",
    )

    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=clean_x,
        yaxis_title=clean_y,
    )
    return _serialize_plotly_fig(fig)


def create_scatter_plot(
    dataset_id: str,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create interactive scatter plot with optional color/size dimensions."""
    df = get_dataset(dataset_id)
    cols = [x, y] + ([color] if color else []) + ([size] if size else [])
    plot_df = df[cols].dropna()

    clean_x = format_label(x)
    clean_y = format_label(y)
    display_title = title or f"{clean_y} vs {clean_x}"

    labels = {x: clean_x, y: clean_y}
    if color:
        labels[color] = format_label(color)
    if size:
        labels[size] = format_label(size)

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        size=size,
        title=display_title,
        color_discrete_sequence=DATAMIND_PALETTE,
        opacity=0.85,
        labels=labels,
    )
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=clean_x,
        yaxis_title=clean_y,
    )
    return _serialize_plotly_fig(fig)


def create_histogram(
    dataset_id: str,
    x: str,
    nbins: int = 30,
    title: str | None = None,
) -> dict[str, Any]:
    """Create interactive histogram for distribution analysis."""
    df = get_dataset(dataset_id)
    clean_x = format_label(x)
    display_title = title or f"Distribution & Spread of {clean_x}"

    fig = px.histogram(
        df,
        x=x,
        nbins=nbins,
        title=display_title,
        color_discrete_sequence=[DATAMIND_PALETTE[1]],
        labels={x: clean_x},
    )
    fig.update_traces(marker=dict(line=dict(color="#111118", width=1)))
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=clean_x,
        yaxis_title="Record Count",
    )
    return _serialize_plotly_fig(fig)


def create_box_plot(
    dataset_id: str,
    y: str,
    x: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create interactive box plot for outlier detection and quantile comparison."""
    df = get_dataset(dataset_id)
    clean_y = format_label(y)
    clean_x = format_label(x) if x else None
    display_title = title or (f"{clean_y} Variance across {clean_x}" if clean_x else f"Box Plot of {clean_y}")

    labels = {y: clean_y}
    if x:
        labels[x] = clean_x

    fig = px.box(
        df,
        x=x,
        y=y,
        title=display_title,
        color_discrete_sequence=DATAMIND_PALETTE,
        labels=labels,
    )
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=clean_x or "",
        yaxis_title=clean_y,
    )
    return _serialize_plotly_fig(fig)


def create_correlation_heatmap(
    dataset_id: str,
    columns: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create interactive correlation heatmap with annotated coefficients and clean labels."""
    df = get_dataset(dataset_id)
    numeric_df = df.select_dtypes(include=[np.number])

    # Filter out ID columns from correlation
    numeric_df = numeric_df[[c for c in numeric_df.columns if not _is_id_column(c, numeric_df[c])]]

    if columns:
        cols = [c for c in columns if c in numeric_df.columns]
        numeric_df = numeric_df[cols]

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for correlation heatmap.")

    corr = numeric_df.corr().round(2)
    clean_labels = [format_label(c) for c in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale=[
                [0.0, "#f87171"],
                [0.5, "#1a1a24"],
                [1.0, "#10b981"],
            ],
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            hoverongaps=False,
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or "Key Feature Correlation Matrix",
        **DARK_LAYOUT,
    )
    return _serialize_plotly_fig(fig)


def create_pie_chart(
    dataset_id: str,
    names: str,
    values: str | None = None,
    title: str | None = None,
    top_n: int = 7,
) -> dict[str, Any]:
    """Create interactive Power BI donut chart for composition & market share."""
    df = get_dataset(dataset_id)
    clean_names = format_label(names)
    clean_values = format_label(values) if values else "Count"

    if values:
        plot_df = df.groupby(names, as_index=False)[values].sum().sort_values(by=values, ascending=False)
    else:
        plot_df = df[names].value_counts().reset_index()
        plot_df.columns = [names, "count"]
        values = "count"

    # Consolidate smaller categories into 'Other' if > top_n
    if len(plot_df) > top_n:
        top_slice = plot_df.head(top_n)
        other_sum = plot_df.iloc[top_n:][values].sum()
        other_row = pd.DataFrame([{names: "Other", values: other_sum}])
        plot_df = pd.concat([top_slice, other_row], ignore_index=True)

    display_title = title or f"{clean_names} Breakdown & Share"

    fig = px.pie(
        plot_df,
        names=names,
        values=values,
        hole=0.62,  # Power BI Donut style
        title=display_title,
        color_discrete_sequence=DATAMIND_PALETTE,
        labels={names: clean_names, values: clean_values},
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="outside",
        marker=dict(line=dict(color="#111118", width=2)),
        opacity=0.92,
        hovertemplate=f"<b>%{{label}}</b><br>{clean_values}: %{{value:,.2f}}<br>Share: %{{percent}}<extra></extra>",
    )
    fig.update_layout(**DARK_LAYOUT)
    return _serialize_plotly_fig(fig)


def recommend_charts_for_dataset(dataset_id: str) -> list[dict[str, Any]]:
    """
    Intelligently select the best charts according to:
      1. Composition (Donut for 2-6 category shares)
      2. Comparison / Ranking (Horizontal/Vertical Bar for multi-categories)
      3. Trend (Spline Area chart for time-series)
      4. Correlation / Relationship (Scatter or Heatmap)
      5. Distribution (Histogram / Density)

    Formats all column names into clean, professional business titles.
    """
    df = get_dataset(dataset_id)
    recommendations = []

    # 1. Filter out ID columns
    valid_cols = [c for c in df.columns if not _is_id_column(c, df[c])]

    # 2. Categorize data types
    dt_cols = []
    for c in valid_cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
        elif df[c].dtype == "object":
            col_l = c.lower()
            if any(k in col_l for k in ["date", "time", "timestamp", "created_at", "order_date", "year", "month"]):
                try:
                    pd.to_datetime(df[c].dropna().head(20))
                    dt_cols.append(c)
                except Exception:
                    pass

    num_cols = [
        c for c in valid_cols 
        if pd.api.types.is_numeric_dtype(df[c]) and c not in dt_cols
    ]
    cat_cols = [
        c for c in valid_cols 
        if c not in num_cols and c not in dt_cols
    ]

    # Prioritize high-importance business metrics
    def metric_priority(name: str) -> int:
        nl = name.lower()
        keywords = ["revenue", "sales", "profit", "amount", "price", "cost", "score", "total", "val", "spend", "rate", "income", "churn", "qty"]
        for idx, kw in enumerate(keywords):
            if kw in nl:
                return idx
        return 99

    num_cols.sort(key=metric_priority)

    # Segregate categories by cardinality
    donut_candidates = []  # 2 to 6 unique values (perfect for composition donut)
    bar_candidates = []    # 3 to 15 unique values (perfect for bar ranking)

    for c in cat_cols:
        n_unique = df[c].nunique()
        if 2 <= n_unique <= 6:
            donut_candidates.append(c)
        if 2 <= n_unique <= 25:
            bar_candidates.append(c)

    primary_metric = num_cols[0] if num_cols else None
    secondary_metric = num_cols[1] if len(num_cols) > 1 else None

    # ─── 1. Primary Visual: Trend or Key Comparison ──────────────────────────
    if dt_cols and primary_metric:
        recommendations.append({
            "chart_type": "line",
            "x": dt_cols[0],
            "y": primary_metric,
            "title": f"{format_label(primary_metric)} Trend over Time",
            "reason": "Temporal trend line with gradient fill",
        })
    elif bar_candidates and primary_metric:
        top_cat = bar_candidates[0]
        # Use horizontal bar if category names are long (> 10 chars average)
        sample_lens = df[top_cat].dropna().astype(str).str.len()
        avg_len = sample_lens.mean() if len(sample_lens) > 0 else 0
        orientation = "h" if avg_len > 9 else "v"

        recommendations.append({
            "chart_type": "bar",
            "x": top_cat,
            "y": primary_metric,
            "orientation": orientation,
            "title": f"{format_label(primary_metric)} by {format_label(top_cat)}",
            "reason": "Executive category ranking with direct data labels",
        })

    # ─── 2. Composition Visual: Donut Chart ─────────────────────────────────
    # Directly satisfies "use proper charts according to the composition"
    if donut_candidates:
        d_cat = donut_candidates[0]
        recommendations.append({
            "chart_type": "pie",
            "names": d_cat,
            "values": primary_metric,
            "title": f"{format_label(d_cat)} Composition & Market Share",
            "reason": "Composition share breakdown using Power BI donut visual",
        })
    elif len(cat_cols) > 0 and primary_metric:
        recommendations.append({
            "chart_type": "pie",
            "names": cat_cols[0],
            "values": primary_metric,
            "title": f"{format_label(cat_cols[0])} Share Breakdown",
            "reason": "Segment composition",
        })

    # ─── 3. Secondary Comparison / Ranking Bar Chart ────────────────────────
    # Choose a different category than the one already used
    used_cats = [r.get("x") or r.get("names") for r in recommendations]
    remaining_bars = [c for c in bar_candidates if c not in used_cats]

    if remaining_bars and primary_metric:
        sec_cat = remaining_bars[0]
        recommendations.append({
            "chart_type": "bar",
            "x": sec_cat,
            "y": primary_metric,
            "orientation": "h",
            "title": f"Top Performing {format_label(sec_cat)} Segments",
            "reason": "Horizontal ranking visual for categorical breakdown",
        })
    elif primary_metric and secondary_metric:
        recommendations.append({
            "chart_type": "scatter",
            "x": primary_metric,
            "y": secondary_metric,
            "title": f"{format_label(secondary_metric)} vs {format_label(primary_metric)} Correlation",
            "reason": "Bivariate scatter correlation",
        })

    # ─── 4. Correlation Heatmap or Distribution ──────────────────────────────
    if len(num_cols) >= 3:
        recommendations.append({
            "chart_type": "correlation_heatmap",
            "columns": num_cols[:7],
            "title": "Inter-Feature Correlation Matrix",
            "reason": "Multi-variable correlation heatmap with clean labels",
        })
    elif primary_metric:
        recommendations.append({
            "chart_type": "histogram",
            "x": primary_metric,
            "title": f"{format_label(primary_metric)} Distribution & Spread",
            "reason": "Univariate frequency and density analysis",
        })

    # Fallback if recommendations are sparse
    if len(recommendations) < 3 and primary_metric:
        recommendations.append({
            "chart_type": "histogram",
            "x": primary_metric,
            "title": f"{format_label(primary_metric)} Distribution Curve",
            "reason": "Density spread",
        })

    return recommendations
