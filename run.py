import ast
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly import colors as plotly_colors
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

st.set_page_config(
    page_title="Peak Explorer",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.0rem;
    }

    div[data-baseweb="select"] span[data-baseweb="tag"] {
        background-color: #edf1f5;
        color: #475467;
        border: 1px solid #d0d5dd;
    }

    div[data-baseweb="select"] span[data-baseweb="tag"] svg {
        fill: #475467;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
BASE_REQUIRED_COLUMNS = {
    "peak_label",
    "ms_file_label",
    "peak_area",
    "peak_area_top3",
    "peak_max",
    "scan_time",
    "intensity",
}

DEFAULT_DATA_PATH = Path(__file__).with_name("results_backup.csv")
DEFAULT_DATA_DESCRIPTION = (
    "This app accepts MINT output files in `.csv` or `.parquet` format. "
    "It expects at least these columns: `peak_label`, `ms_file_label`, "
    "`peak_area`, `peak_area_top3`, `peak_max`, `scan_time`, and "
    "`intensity`.\n\n"
    "If you do not upload a file, the app loads the bundled "
    "`results_backup.csv` example dataset so you can explore the interface "
    "immediately."
)


def to_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return np.nan
    return x


def get_samples(df, peak_label):
    return sorted(
        df.loc[df["peak_label"].astype(str) == str(peak_label), "ms_file_label"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


@st.cache_data(show_spinner=False)
def build_exploded(df, compound):
    sub = df[df["peak_label"].astype(str) == str(compound)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ms_file_label", "scan_time", "intensity"])

    sub["scan_time"] = sub["scan_time"].apply(to_list)
    sub["intensity"] = sub["intensity"].apply(to_list)

    rows = []
    for _, row in sub.iterrows():
        st_vals = row["scan_time"]
        int_vals = row["intensity"]

        if not isinstance(st_vals, (list, tuple, np.ndarray, pd.Series)):
            continue
        if not isinstance(int_vals, (list, tuple, np.ndarray, pd.Series)):
            continue

        for t, y in zip(st_vals, int_vals):
            rows.append({
                "ms_file_label": str(row["ms_file_label"]),
                "scan_time": t,
                "intensity": y,
            })

    exploded = pd.DataFrame(rows)
    if exploded.empty:
        return pd.DataFrame(columns=["ms_file_label", "scan_time", "intensity"])

    exploded["scan_time"] = pd.to_numeric(exploded["scan_time"], errors="coerce")
    exploded["intensity"] = pd.to_numeric(exploded["intensity"], errors="coerce")
    exploded = exploded.dropna(subset=["ms_file_label", "scan_time", "intensity"])
    return exploded.sort_values(["ms_file_label", "scan_time"])


def choose_scale(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 1, ""

    vmax = values.abs().max()

    if vmax >= 1e9:
        return 1e9, "x 10^9"
    if vmax >= 1e6:
        return 1e6, "x 10^6"
    if vmax >= 1e3:
        return 1e3, "x 10^3"
    return 1, ""


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is None:
        if DEFAULT_DATA_PATH.exists():
            return pd.read_csv(DEFAULT_DATA_PATH)
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    raise ValueError("Unsupported file format. Upload a CSV or Parquet file.")


def validate_df(df):
    missing = BASE_REQUIRED_COLUMNS - set(df.columns)
    return sorted(missing)


def get_sample_colors(selected_samples):
    palette = plotly_colors.qualitative.Plotly
    return {
        str(sample): palette[i % len(palette)]
        for i, sample in enumerate(selected_samples)
    }


def render_sample_color_key(selected_samples, sample_colors):
    if not selected_samples:
        return

    chips = []
    for sample in selected_samples:
        color = sample_colors[str(sample)]
        chips.append(
            (
                "<span style='display:inline-flex;align-items:center;gap:0.4rem;"
                "margin:0 0.5rem 0.5rem 0;padding:0.2rem 0.55rem;border-radius:999px;"
                "background:#f3f4f6;border:1px solid #e5e7eb;font-size:0.9rem;'>"
                f"<span style='width:0.75rem;height:0.75rem;border-radius:999px;"
                f"background:{color};display:inline-block;'></span>"
                f"{sample}</span>"
            )
        )

    st.markdown("".join(chips), unsafe_allow_html=True)


def build_figure_title(compound, selected_samples, max_chars=90):
    if not selected_samples:
        return str(compound)

    sample_text = ", ".join(str(sample) for sample in selected_samples)
    full_title = f"{compound}  •  {sample_text}"
    if len(full_title) <= max_chars:
        return full_title

    if len(selected_samples) == 1:
        clipped = str(selected_samples[0])[: max(12, max_chars - len(str(compound)) - 8)].rstrip()
        return f"{compound}  •  {clipped}..."

    return f"{compound}  •  {len(selected_samples)} selected samples"


def make_figure(df, compound, selected_samples, metric):
    left_x_domain = [0.0, 0.43]
    right_x_domain = [0.57, 1.0]
    y_domain = [0.0, 1.0]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{metric} distribution", "Chromatograms"),
        horizontal_spacing=0.14,
    )

    sample_colors = get_sample_colors(selected_samples)

    # -----------------------------
    # Left panel: peak area distribution
    # -----------------------------
    sub = df[df["peak_label"].astype(str) == str(compound)].copy()

    peak_summary = sub.groupby("ms_file_label", dropna=True)[metric].median().reset_index()
    peak_summary[metric] = pd.to_numeric(peak_summary[metric], errors="coerce")
    peak_summary = peak_summary.dropna(subset=[metric])

    scale, scale_label = choose_scale(peak_summary[metric])
    peak_summary["scaled"] = peak_summary[metric] / scale

    if len(peak_summary) >= 3 and peak_summary["scaled"].nunique() > 1:
        kde_values = peak_summary["scaled"].to_numpy(dtype=float)
        kde = gaussian_kde(kde_values)
        x_max = max(float(kde_values.max()) * 1.08, 1.0)
        x_grid = np.linspace(0, x_max, 256)
        y_grid = kde(x_grid)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                line=dict(color="#8b9098", width=2),
                fill="tozeroy",
                fillcolor="rgba(139, 144, 152, 0.25)",
                hovertemplate=f"{metric}: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
                name="KDE",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        y_max = float(np.nanmax(y_grid)) if len(y_grid) else 1.0
    else:
        y_max = 1.0
        if len(peak_summary) > 0:
            fig.add_trace(
                go.Scatter(
                    x=peak_summary["scaled"],
                    y=np.zeros(len(peak_summary)),
                    mode="markers",
                    marker=dict(color="#9aa0a6", size=9),
                    hovertemplate=f"sample: %{{customdata}}<br>{metric}: %{{x:.3f}}<extra></extra>",
                    customdata=peak_summary["ms_file_label"],
                    name=f"{metric} values",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                text=f"No {metric} data",
                showarrow=False,
                font=dict(size=12, color="#6b7280"),
                row=1,
                col=1,
            )

    for i, sample in enumerate(selected_samples):
        val = peak_summary.loc[
            peak_summary["ms_file_label"].astype(str) == str(sample), "scaled"
        ]
        if not val.empty:
            x_val = float(val.iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],
                    y=[0, y_max],
                    mode="lines",
                    line=dict(color=sample_colors[str(sample)], width=2, dash="dash"),
                    hovertemplate=f"sample: {sample}<br>{metric}: %{{x:.3f}}<extra></extra>",
                    name=sample,
                    legendgroup=str(sample),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # -----------------------------
    # Right panel: chromatograms
    # -----------------------------
    exploded = build_exploded(df, compound)
    chromatogram_y_max = 1.0

    if not exploded.empty:
        background_samples = [
            sample_name
            for sample_name in exploded["ms_file_label"].astype(str).drop_duplicates().tolist()
            if str(sample_name) not in {str(sample) for sample in selected_samples}
        ]
        chromatogram_y_max = max(float(exploded["intensity"].max()), 1.0)

        for i, sample_name in enumerate(background_samples):
            grp = exploded[exploded["ms_file_label"].astype(str) == str(sample_name)].copy()
            grp = grp.sort_values("scan_time")
            fig.add_trace(
                go.Scatter(
                    x=grp["scan_time"],
                    y=grp["intensity"],
                    mode="lines",
                    line=dict(color="rgba(128, 128, 128, 0.22)", width=1),
                    hovertemplate="sample: %{meta}<br>scan_time: %{x:.4f}<br>intensity: %{y:.4f}<extra></extra>",
                    meta=str(sample_name),
                    name="Other samples",
                    legendgroup="other_samples",
                    showlegend=(i == 0),
                ),
                row=1,
                col=2,
            )

        for i, sample in enumerate(selected_samples):
            sel = exploded[exploded["ms_file_label"].astype(str) == str(sample)].copy()
            sel = sel.sort_values("scan_time")
            if not sel.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sel["scan_time"],
                        y=sel["intensity"],
                        mode="lines",
                        line=dict(color=sample_colors[str(sample)], width=2.5),
                        hovertemplate="sample: %{meta}<br>scan_time: %{x:.4f}<br>intensity: %{y:.4f}<extra></extra>",
                        meta=str(sample),
                        name=str(sample),
                        legendgroup=str(sample),
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x2 domain",
            yref="y2 domain",
            text="No chromatogram data",
            showarrow=False,
            font=dict(size=12, color="#6b7280"),
        )

    x_title = f"{metric} ({scale_label})" if scale_label else metric
    density_title = "Density" if len(peak_summary) >= 3 and peak_summary["scaled"].nunique() > 1 else ""
    fig.update_xaxes(
        title_text=x_title,
        rangemode="tozero",
        domain=left_x_domain,
        title_standoff=22,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=density_title,
        range=[0, max(y_max * 1.05, 1e-6)],
        domain=y_domain,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Scan Time",
        domain=right_x_domain,
        title_standoff=22,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Intensity",
        range=[0, chromatogram_y_max * 1.05],
        domain=y_domain,
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=55, r=55, t=125, b=75),
        height=520,
        legend=dict(
            title_text="Displayed traces",
            x=1.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        legend_groupclick="togglegroup",
        hovermode="closest",
    )

    fig.add_annotation(
        x=0.5,
        y=1.26,
        xref="paper",
        yref="paper",
        text=f"<b>{build_figure_title(compound, selected_samples)}</b>",
        showarrow=False,
        font=dict(size=17, color="#24324b"),
    )

    return fig


# -----------------------------
# App
# -----------------------------
st.title("Peak Explorer")

uploaded_file = st.sidebar.file_uploader(
    "Upload data",
    type=["csv", "parquet"],
    help="Upload a CSV or Parquet file containing peak_label, ms_file_label, peak_area, peak_area_top3, scan_time, and intensity. If no file is uploaded, the app uses the bundled results_backup.csv."
)
st.sidebar.markdown(DEFAULT_DATA_DESCRIPTION)

df = load_data(uploaded_file)

if df is None:
    st.info("Upload a CSV or Parquet file to begin.")
    st.stop()

if uploaded_file is None:
    st.caption("Currently viewing the bundled example dataset: `results_backup.csv`.")
    st.sidebar.markdown(f"**Using bundled data:** `{DEFAULT_DATA_PATH.name}`")

missing = validate_df(df)
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

compound_options = sorted(df["peak_label"].dropna().astype(str).unique().tolist())
if not compound_options:
    st.error("No valid values found in peak_label.")
    st.stop()

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    compound = st.selectbox("Compound", compound_options)

with c2:
    metric = st.selectbox("Metric", ["peak_area", "peak_area_top3", "peak_max"], index=1)

sample_options = get_samples(df, compound)

with c3:
    selected_samples = st.multiselect(
        "Samples",
        options=sample_options,
        default=sample_options[:1] if sample_options else [],
    )
    render_sample_color_key(selected_samples, get_sample_colors(selected_samples))

if not selected_samples:
    st.warning("Select at least one sample.")
    st.stop()

fig = make_figure(df, compound, selected_samples, metric)
st.plotly_chart(fig, width="stretch")

preview_df = df[
    (df["peak_label"].astype(str) == str(compound))
    & (df["ms_file_label"].astype(str).isin([str(sample) for sample in selected_samples]))
].copy()

with st.expander("Preview data"):
    st.caption("Rows matching the selected compound and samples.")
    st.dataframe(preview_df.head(50), width="stretch")
