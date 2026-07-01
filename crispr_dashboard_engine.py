#!/usr/bin/env python3
"""Interactive CRISPR screen analysis dashboard (Streamlit + Plotly WebGL).

A single-file, multi-tab web application that turns a static ``sgRNA`` enrichment
table (the output of MAGeCK / DESeq2, or of the companion
``generate_synthetic_crispr_screen.py``) into an interactive analysis surface.
It replaces the "open the CSV, run an R script, stare at a static PDF" loop with
live filtering, WebGL-accelerated plots, and one-click export.

Architecture
------------
The app is deliberately modular:

* :func:`load_dataframe` is wrapped in ``@st.cache_data`` so the (potentially
  large) file is parsed and its derived columns computed **once**. Moving a
  slider re-runs the script top-to-bottom, but the cached frame is returned
  instantly -- only the cheap in-memory filtering re-executes.
* Every plot is built by its own pure ``build_*`` function that takes a frame and
  returns a ``plotly`` figure, so the rendering logic is testable and reusable.
* All scatter layers use :class:`plotly.graph_objects.Scattergl` (WebGL), which
  keeps a 15k-point volcano plot smooth where the SVG renderer would stutter.

Run
---
::

    streamlit run crispr_dashboard_engine.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_FILE: str = "processed_sgRNA_enrichment.csv"
DEFAULT_NTC_LABEL: str = "NTC"

# Columns the pipeline must provide. Validated on load so a malformed upload
# fails loudly instead of erroring deep inside a plot call.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "sgRNA_ID",
    "Target_Gene",
    "Base_Mean_Counts",
    "LFC_Treatment_vs_Control",
    "p_value",
    "FDR",
    "Pathway_Annotation",
)

# Accepted source-column names per canonical column, so uploads from real
# pipelines (MAGeCK, DESeq2, edgeR/limma) map automatically without the user
# renaming anything. Matching is case-insensitive and whitespace-trimmed. Keep
# every alias lower-cased here. ``Pathway_Annotation`` is intentionally optional
# (many count pipelines don't emit it) and is synthesised when absent.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sgRNA_ID": ("sgrna_id", "sgrna", "sgrna_name", "guide", "guide_id",
                 "grna", "barcode", "id"),
    "Target_Gene": ("target_gene", "gene", "gene_symbol", "genesymbol",
                    "symbol", "gene_id", "geneid"),
    "Base_Mean_Counts": ("base_mean_counts", "basemean", "base_mean",
                         "basemeancounts", "meancounts", "mean_counts",
                         "mean_count", "avgcount", "abundance"),
    "LFC_Treatment_vs_Control": ("lfc_treatment_vs_control", "lfc",
                                 "log2foldchange", "log2fc", "logfc",
                                 "log2_fold_change", "neg|lfc"),
    "p_value": ("p_value", "pvalue", "p.value", "p-value", "p.twosided",
                "pval", "neg|p-value", "neg|p.value"),
    "FDR": ("fdr", "padj", "adj.p.val", "qvalue", "q_value", "p.adjust",
            "neg|fdr", "adjusted_pvalue", "adj_pvalue"),
    "Pathway_Annotation": ("pathway_annotation", "pathway", "annotation",
                           "kegg", "go", "geneset", "gene_set", "term"),
}

# Semantic classification labels shared across every tab.
CAT_NTC: str = "Non-Targeting Control"
CAT_NONSIG: str = "Not Significant"
CAT_UP: str = "Enriched (Up)"
CAT_DOWN: str = "Depleted (Down)"
CAT_HIGHLIGHT: str = "Search Highlight"

# Colour language. Red = enriched dependency loss / resistance, Blue = depleted
# (essential) -- the field-standard volcano palette. NTCs are a faint grey cloud;
# non-significant genes are muted so the eye locks onto real hits.
COLORS: Dict[str, str] = {
    CAT_NTC: "rgba(140, 140, 140, 0.35)",
    CAT_NONSIG: "rgba(184, 184, 184, 0.55)",
    CAT_UP: "rgba(214, 39, 40, 0.85)",
    CAT_DOWN: "rgba(31, 91, 176, 0.85)",
    CAT_HIGHLIGHT: "rgba(255, 214, 0, 1.0)",
}

# Small numeric floor so FDR == 0 (float underflow on huge effects) maps to a
# finite -log10 value instead of +inf.
FDR_FLOOR: float = 1e-300

# Discrete options for the log-scaled FDR slider (a linear slider over an
# exponential quantity is unusable; a curated ladder matches how analysts think).
FDR_THRESHOLD_OPTIONS: Tuple[float, ...] = (
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 1.0,
)
COUNT_CUTOFF_OPTIONS: Tuple[int, ...] = (0, 10, 25, 50, 100, 200, 500, 1000)

# --- Automated QC thresholds (biologically motivated defaults) ---------------
NTC_SPREAD_WARN: float = 0.35        # SD of NTC LFC above this == noisy control cloud.
NTC_SPREAD_FAIL: float = 0.50        # ...and above this it's a real problem.
NTC_BIAS_WARN: float = 0.15          # |median NTC LFC| above this == off-centre.
DROPOUT_READ_FLOOR: int = 30         # Guides below this are "dropout" candidates.
DROPOUT_FRAC_WARN: float = 0.15      # >15% low-read guides == representation problem.
ESSENTIAL_DEPLETION_PASS: float = -1.0   # Median essential LFC this low == real signal.
ESSENTIAL_DEPLETION_WARN: float = -0.5   # ...between here and PASS is a weak signal.

# Canonical pan-essential "positive control" panel (ribosome / proteasome /
# RNA Pol II / spliceosome). Any working screen MUST deplete these; if it doesn't,
# the screen failed or the treatment/control labels are swapped. In production you
# would swap this for the full Hart et al. reference core-essential gene list.
CORE_ESSENTIAL_CONTROLS: Tuple[str, ...] = (
    "RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPS3", "RPS6", "RPS11",
    "RPS14", "RPS18", "POLR2A", "POLR2B", "POLR2C", "PSMA1", "PSMB2",
    "PSMC1", "PSMD1", "EIF4A3", "SF3B1", "SNRPD1",
)


@dataclass
class QCFlag:
    """A single automated quality-control assessment.

    Attributes:
        status: Severity, one of ``"pass"``, ``"warn"``, ``"fail"``, ``"info"``.
        title: Short bold headline for the flag.
        detail: One-line human explanation including the numbers that drove it.
    """

    status: str
    title: str
    detail: str


# ---------------------------------------------------------------------------
# Data ingestion (cached)
# ---------------------------------------------------------------------------
def _harmonize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised source columns to the app's canonical schema.

    Lets an analyst drop a raw MAGeCK / DESeq2 / edgeR table straight in without
    editing headers: aliases in :data:`COLUMN_ALIASES` are matched
    case-insensitively and renamed. Missing ``Pathway_Annotation`` is synthesised
    as ``"Unassigned"``, and a missing ``Base_Mean_Counts`` is derived from paired
    per-condition count columns (the MAGeCK sgRNA-summary layout) when present.

    Args:
        frame: The raw table exactly as read from the uploaded/default file.

    Returns:
        pd.DataFrame: The same rows with columns renamed to the canonical schema
        wherever a confident match was found.
    """
    lower_to_actual = {str(c).lower().strip(): c for c in frame.columns}
    rename: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue  # Already canonical -- never rename over an exact match.
        for alias in aliases:
            actual = lower_to_actual.get(alias)
            if actual is not None and actual not in rename:
                rename[actual] = canonical
                break
    if rename:
        frame = frame.rename(columns=rename)

    # Pathway annotation is optional in most count pipelines; synthesise a neutral
    # column so every downstream tab can rely on a uniform schema.
    if "Pathway_Annotation" not in frame.columns:
        frame["Pathway_Annotation"] = "Unassigned"

    # Derive a base mean from paired condition counts if the pipeline (e.g. a
    # MAGeCK sgRNA summary) reports per-arm counts instead of a single baseMean.
    if "Base_Mean_Counts" not in frame.columns:
        lower_to_actual = {str(c).lower().strip(): c for c in frame.columns}
        ctrl = lower_to_actual.get("control_count") or lower_to_actual.get("control_mean")
        treat = lower_to_actual.get("treatment_count") or lower_to_actual.get("treat_count")
        if ctrl and treat:
            ctrl_num = pd.to_numeric(frame[ctrl], errors="coerce")
            treat_num = pd.to_numeric(frame[treat], errors="coerce")
            if ctrl_num.notna().any() and treat_num.notna().any():
                frame["Base_Mean_Counts"] = (ctrl_num.fillna(0) + treat_num.fillna(0)) / 2.0

    return frame


@st.cache_data(show_spinner="Parsing enrichment table...")
def load_dataframe(
    file_bytes: Optional[bytes],
    filename: Optional[str],
    default_path: str,
) -> pd.DataFrame:
    """Load, validate and enrich the sgRNA enrichment table (cached).

    Reading and column-derivation happen exactly once per unique input thanks to
    ``@st.cache_data`` -- Streamlit hashes ``file_bytes``/``filename``/
    ``default_path``, so subsequent slider interactions reuse the parsed frame
    instead of re-reading from disk on every rerun.

    Args:
        file_bytes: Raw bytes of an uploaded file, or ``None`` to fall back to
            the bundled default dataset.
        filename: Original name of the uploaded file (used only to sniff the
            delimiter). ``None`` when loading the default.
        default_path: Path to the default CSV shipped alongside the app.

    Returns:
        pd.DataFrame: The validated table with two appended helper columns,
        ``neg_log10_FDR`` and ``log10_Base_Mean``.

    Raises:
        FileNotFoundError: If no upload is given and the default file is absent.
        ValueError: If any required column is missing from the input.
    """
    if file_bytes is not None:
        # TSV/TXT are tab-delimited; everything else is treated as comma-sep.
        sep = "\t" if filename and filename.lower().endswith((".tsv", ".txt")) else ","
        frame = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    else:
        path = Path(default_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Default dataset '{default_path}' not found. Generate it first "
                f"with: python generate_synthetic_crispr_screen.py"
            )
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        frame = pd.read_csv(path, sep=sep)

    # Auto-map real-pipeline headers (MAGeCK/DESeq2/edgeR) to our canonical schema
    # before validating, so a raw export uploads without any manual renaming.
    frame = _harmonize_columns(frame)

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "Could not map required column(s) after alias matching: "
            + ", ".join(missing)
            + ".\n\nColumns found in your file: "
            + ", ".join(map(str, frame.columns))
            + f".\n\nExpected canonical schema: {', '.join(REQUIRED_COLUMNS)}."
            + "\nRename the offending column(s) to a canonical name, or extend "
            + "COLUMN_ALIASES with your pipeline's header."
        )

    # Coerce the numeric columns defensively (uploads sometimes carry stray text).
    for col in ("Base_Mean_Counts", "LFC_Treatment_vs_Control", "p_value", "FDR"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["LFC_Treatment_vs_Control", "FDR", "Base_Mean_Counts"])

    # Pre-compute the two transforms every plot needs, once.
    frame["neg_log10_FDR"] = -np.log10(frame["FDR"].clip(lower=FDR_FLOOR))
    frame["log10_Base_Mean"] = np.log10(frame["Base_Mean_Counts"].clip(lower=1))
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classification (cheap, runs every rerun on the cached frame)
# ---------------------------------------------------------------------------
def classify_guides(
    frame: pd.DataFrame,
    fdr_threshold: float,
    lfc_threshold: float,
    count_cutoff: int,
    ntc_label: str,
) -> pd.Series:
    """Assign each guide to a display category given the active filter settings.

    A guide is a *significant hit* when it simultaneously clears three gates:
    FDR below threshold, absolute LFC at/above threshold, and base-mean counts
    at/above the low-read cutoff (guides below the read floor are statistically
    untrustworthy and are demoted to "not significant" regardless of their LFC).

    Args:
        frame: The loaded enrichment table.
        fdr_threshold: Maximum FDR for a guide to count as significant.
        lfc_threshold: Minimum ``|LFC|`` for a guide to count as significant.
        count_cutoff: Minimum ``Base_Mean_Counts`` to be eligible at all.
        ntc_label: The value in ``Target_Gene`` that marks non-targeting controls.

    Returns:
        pd.Series: Categorical label per guide (one of the ``CAT_*`` constants),
        aligned to ``frame``.
    """
    lfc = frame["LFC_Treatment_vs_Control"].to_numpy()
    fdr = frame["FDR"].to_numpy()
    counts = frame["Base_Mean_Counts"].to_numpy()
    is_ntc = frame["Target_Gene"].to_numpy() == ntc_label

    passes = (fdr < fdr_threshold) & (np.abs(lfc) >= lfc_threshold) & (counts >= count_cutoff)

    category = np.full(len(frame), CAT_NONSIG, dtype=object)
    category[passes & (lfc > 0)] = CAT_UP
    category[passes & (lfc < 0)] = CAT_DOWN
    # NTCs are labelled last so they always render as controls, never as "hits",
    # even on the rare occasion a control crosses threshold by chance.
    category[is_ntc] = CAT_NTC
    return pd.Series(category, index=frame.index, name="Category")


def parse_search_terms(raw: str) -> Set[str]:
    """Normalise a free-text gene search box into a set of upper-cased symbols.

    Accepts comma-, space- or newline-separated gene symbols so an analyst can
    paste a small panel (e.g. ``TP53, EGFR MYC``) in one go.

    Args:
        raw: The raw contents of the search text input.

    Returns:
        Set[str]: Upper-cased, whitespace-stripped, de-duplicated search terms.
    """
    if not raw:
        return set()
    tokens = raw.replace(",", " ").replace("\n", " ").split()
    return {token.strip().upper() for token in tokens if token.strip()}


# ---------------------------------------------------------------------------
# Automated QC intelligence ("grad-student-in-a-box")
# ---------------------------------------------------------------------------
def compute_qc_flags(frame: pd.DataFrame, ntc_label: str) -> List[QCFlag]:
    """Turn raw QC numbers into explained biological verdicts.

    Encodes the sanity checks an experienced screener performs by eye into
    explicit, self-explaining flags:

    1. **NTC spread** -- is the technical-noise floor tight? A wide control cloud
       points to a library-representation / normalisation problem.
    2. **NTC centring** -- do the controls sit on zero? An off-zero median means
       the normalisation is biased.
    3. **Dropout** -- what fraction of the library sits below the read floor? A
       heavy low-read shoulder makes those guides untrustworthy.
    4. **Positive control** -- did the pan-essential genes actually deplete? This
       is the single most important "did my screen work?" check; a failure here
       usually means the treatment/control labels are swapped.

    Args:
        frame: The loaded enrichment table.
        ntc_label: The value in ``Target_Gene`` marking non-targeting controls.

    Returns:
        List[QCFlag]: Ordered assessments, most-fundamental check first.
    """
    flags: List[QCFlag] = []
    lfc = "LFC_Treatment_vs_Control"

    # 1 & 2) Non-targeting controls: spread (noise floor) and centring (bias).
    ntc = frame.loc[frame["Target_Gene"] == ntc_label, lfc]
    if ntc.empty:
        flags.append(QCFlag(
            "info", "No controls found",
            f"No guides labelled '{ntc_label}'. Set the correct control label in "
            "the sidebar to enable normalisation QC.",
        ))
    else:
        sd = float(ntc.std())
        if sd >= NTC_SPREAD_FAIL:
            flags.append(QCFlag(
                "fail", "NTC spread very high",
                f"Control LFC SD = {sd:.3f} (fail ≥ {NTC_SPREAD_FAIL}). Strong "
                "evidence of a library-representation or normalisation problem.",
            ))
        elif sd >= NTC_SPREAD_WARN:
            flags.append(QCFlag(
                "warn", "NTC spread elevated",
                f"Control LFC SD = {sd:.3f} (warn ≥ {NTC_SPREAD_WARN}). Possible "
                "representation issue — inspect the low-count guides.",
            ))
        else:
            flags.append(QCFlag(
                "pass", "NTC spread within range",
                f"Control LFC SD = {sd:.3f}. Technical noise looks normal.",
            ))

        med = float(ntc.median())
        if abs(med) >= NTC_BIAS_WARN:
            flags.append(QCFlag(
                "warn", "Controls off-centre",
                f"Median control LFC = {med:+.3f} (warn |·| ≥ {NTC_BIAS_WARN}). "
                "Normalisation may be biased; NTCs should centre on 0.",
            ))
        else:
            flags.append(QCFlag(
                "pass", "Controls centred on zero",
                f"Median control LFC = {med:+.3f}. Normalisation looks correct.",
            ))

    # 3) Library dropout / representation.
    frac_low = float((frame["Base_Mean_Counts"] < DROPOUT_READ_FLOOR).mean())
    if frac_low >= DROPOUT_FRAC_WARN:
        flags.append(QCFlag(
            "warn", "Library dropout detected",
            f"{frac_low:.1%} of guides under {DROPOUT_READ_FLOOR} reads "
            f"(warn ≥ {DROPOUT_FRAC_WARN:.0%}). Uneven representation or heavy "
            "selection — low-read hits are unreliable.",
        ))
    else:
        flags.append(QCFlag(
            "pass", "Read depth healthy",
            f"Only {frac_low:.1%} of guides under {DROPOUT_READ_FLOOR} reads. "
            "Library representation looks even.",
        ))

    # 4) Positive control: do the pan-essential genes deplete as they must?
    ctrl_genes = frame[frame["Target_Gene"].isin(CORE_ESSENTIAL_CONTROLS)]
    if ctrl_genes.empty:
        flags.append(QCFlag(
            "info", "No essential controls detected",
            "None of the reference pan-essential genes were found, so screen "
            "signal cannot be auto-verified (controls absent or renamed).",
        ))
    else:
        med_ess = float(ctrl_genes[lfc].median())
        n_ess = int(ctrl_genes["Target_Gene"].nunique())
        if med_ess <= ESSENTIAL_DEPLETION_PASS:
            flags.append(QCFlag(
                "pass", "Screen signal confirmed",
                f"{n_ess} core-essential genes depleted (median LFC = {med_ess:.2f}). "
                "Positive controls behaved — the screen has real signal.",
            ))
        elif med_ess <= ESSENTIAL_DEPLETION_WARN:
            flags.append(QCFlag(
                "warn", "Weak screen signal",
                f"Essentials only mildly depleted (median LFC = {med_ess:.2f}). "
                "Selection pressure or coverage may be low.",
            ))
        else:
            flags.append(QCFlag(
                "fail", "Positive controls FAILED",
                f"Essential genes not depleted (median LFC = {med_ess:.2f}). Check "
                "that treatment/control labels aren't swapped — the screen may "
                "not have worked.",
            ))

    return flags


# Maps a flag severity to the Streamlit callout used to render it.
_QC_RENDERERS = {
    "pass": (st.success, "✅"),
    "warn": (st.warning, "⚠️"),
    "fail": (st.error, "\U0001f6d1"),
    "info": (st.info, "ℹ️"),
}
# Severity ordering for computing the single overall verdict (lower == worse).
_QC_SEVERITY = {"fail": 0, "warn": 1, "info": 2, "pass": 3}


def render_qc_flags(flags: List[QCFlag]) -> None:
    """Render the automated QC panel: one overall verdict then each flag.

    Args:
        flags: The assessments from :func:`compute_qc_flags`.
    """
    worst = min((f.status for f in flags), key=lambda s: _QC_SEVERITY[s]) if flags else "pass"
    if worst == "fail":
        st.error("**Overall: screen needs review** — one or more checks failed.")
    elif worst == "warn":
        st.warning("**Overall: passed with cautions** — review the flags below.")
    elif worst == "info":
        st.info("**Overall: incomplete** — some checks could not run (see below).")
    else:
        st.success("**Overall: all automated QC checks passed.**")

    for flag in flags:
        render, icon = _QC_RENDERERS[flag.status]
        render(f"{icon}  **{flag.title}.**  {flag.detail}")


# ---------------------------------------------------------------------------
# Figure builders (pure functions: frame -> go.Figure)
# ---------------------------------------------------------------------------
def _hover_customdata(frame: pd.DataFrame) -> np.ndarray:
    """Stack the per-point hover fields into a Plotly ``customdata`` matrix.

    Args:
        frame: Any subset of the enrichment table.

    Returns:
        np.ndarray: Object array of shape ``(n, 4)`` holding ``sgRNA_ID``,
        ``Target_Gene``, ``FDR`` and ``Base_Mean_Counts`` for each row.
    """
    return np.column_stack(
        [
            frame["sgRNA_ID"].to_numpy(),
            frame["Target_Gene"].to_numpy(),
            frame["FDR"].to_numpy(),
            frame["Base_Mean_Counts"].to_numpy(),
        ]
    )


# Shared hover card used by the MA and volcano plots.
_HOVER_TEMPLATE: str = (
    "<b>%{customdata[0]}</b><br>"
    "Gene: %{customdata[1]}<br>"
    "LFC: %{x:.3f}<br>"
    "FDR: %{customdata[2]:.2e}<br>"
    "Base Mean: %{customdata[3]:,.0f}"
    "<extra></extra>"
)


def build_ma_plot(frame: pd.DataFrame, categories: pd.Series,
                  ntc_label: str) -> go.Figure:
    """Build the MA plot: library representation (x) vs. fold change (y).

    The MA plot is the primary normalisation QC. If the pipeline normalised
    correctly, the bulk of guides -- and *especially* the non-targeting controls
    -- form a flat band centred on ``LFC = 0`` across the whole depth range. A
    tilt or an off-zero NTC cloud signals a normalisation problem. Real hits peel
    off that band symmetrically (blue down, red up).

    Args:
        frame: The loaded enrichment table.
        categories: Per-guide category labels from :func:`classify_guides`.
        ntc_label: The label used for non-targeting controls (for the legend).

    Returns:
        go.Figure: A WebGL MA-plot figure with a dashed reference line at 0.
    """
    fig = go.Figure()

    # Draw order matters: background band first, hits next, controls on top so the
    # normalisation cloud is never hidden behind the bulk of neutral guides.
    layer_order = [CAT_NONSIG, CAT_UP, CAT_DOWN, CAT_NTC]
    legend_names = {CAT_NTC: f"{ntc_label} (control)"}
    for cat in layer_order:
        sub = frame[categories == cat]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["Base_Mean_Counts"],
                y=sub["LFC_Treatment_vs_Control"],
                mode="markers",
                name=legend_names.get(cat, cat),
                marker=dict(
                    color=COLORS[cat],
                    size=6 if cat in (CAT_UP, CAT_DOWN, CAT_NTC) else 4,
                    line=dict(width=0),
                ),
                customdata=_hover_customdata(sub),
                hovertemplate=_HOVER_TEMPLATE,
            )
        )

    # Zero reference: where a perfectly normalised, no-effect guide should sit.
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.45)")
    fig.update_layout(
        title="MA Plot — Library Representation vs. Log2 Fold Change",
        xaxis=dict(title="Base Mean Counts (log scale)", type="log"),
        yaxis=dict(title="Log2 Fold Change (Treatment vs Control)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=520,
        margin=dict(l=60, r=30, t=70, b=50),
        template="plotly_white",
    )
    return fig


def build_count_histogram(frame: pd.DataFrame, ntc_label: str) -> go.Figure:
    """Build the count-distribution histogram (depth / dropout QC).

    Plotted on a log10 base-mean axis and split into targeting vs. control
    guides. A healthy library is roughly bell-shaped in log space; a heavy left
    shoulder piling up at the read floor indicates severe dropout (guides lost
    from the library), while an over-dispersed, multi-modal shape points to
    uneven representation.

    Args:
        frame: The loaded enrichment table.
        ntc_label: The label used for non-targeting controls.

    Returns:
        go.Figure: An overlaid histogram of targeting vs. control read depth.
    """
    is_ntc = frame["Target_Gene"] == ntc_label
    targeting = frame.loc[~is_ntc, "log10_Base_Mean"]
    controls = frame.loc[is_ntc, "log10_Base_Mean"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=targeting, name="Targeting guides", nbinsx=60,
            marker_color="rgba(31, 91, 176, 0.65)",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=controls, name=f"{ntc_label} controls", nbinsx=60,
            marker_color="rgba(214, 39, 40, 0.70)",
        )
    )
    fig.update_layout(
        title="Read-Depth Distribution — Dropout & Over-dispersion Check",
        xaxis_title="log10(Base Mean Counts)",
        yaxis_title="Number of guides",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=520,
        margin=dict(l=60, r=30, t=70, b=50),
        template="plotly_white",
    )
    fig.update_traces(opacity=0.75)
    return fig


def build_volcano_plot(
    frame: pd.DataFrame,
    categories: pd.Series,
    search_terms: Set[str],
) -> go.Figure:
    """Build the high-performance WebGL volcano plot (the hit-discovery engine).

    X = log2 fold change, Y = ``-log10(FDR)``. One ``Scattergl`` trace per
    category keeps the legend interactive (click to toggle) while staying on the
    GPU-accelerated renderer. If the analyst has typed one or more gene symbols,
    every matching guide is redrawn on top as a bright-yellow diamond so a
    specific gene's guides can be located instantly within the cloud.

    Args:
        frame: The loaded enrichment table.
        categories: Per-guide category labels from :func:`classify_guides`.
        search_terms: Upper-cased gene symbols to highlight (may be empty).

    Returns:
        go.Figure: The interactive volcano figure.
    """
    fig = go.Figure()

    # Background-to-foreground draw order for legible layering.
    layer_order = [CAT_NONSIG, CAT_NTC, CAT_DOWN, CAT_UP]
    for cat in layer_order:
        sub = frame[categories == cat]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["LFC_Treatment_vs_Control"],
                y=sub["neg_log10_FDR"],
                mode="markers",
                name=cat,
                marker=dict(
                    color=COLORS[cat],
                    size=7 if cat in (CAT_UP, CAT_DOWN) else 5,
                    line=dict(width=0),
                ),
                customdata=_hover_customdata(sub),
                hovertemplate=_HOVER_TEMPLATE,
            )
        )

    # Search overlay: matching guides painted bright yellow, drawn last (on top).
    if search_terms:
        gene_upper = frame["Target_Gene"].str.upper()
        hit = frame[gene_upper.isin(search_terms)]
        if not hit.empty:
            fig.add_trace(
                go.Scattergl(
                    x=hit["LFC_Treatment_vs_Control"],
                    y=hit["neg_log10_FDR"],
                    mode="markers",
                    name=CAT_HIGHLIGHT,
                    marker=dict(
                        color=COLORS[CAT_HIGHLIGHT],
                        size=12,
                        symbol="diamond",
                        line=dict(width=1.2, color="rgba(60,60,0,0.9)"),
                    ),
                    customdata=_hover_customdata(hit),
                    hovertemplate=_HOVER_TEMPLATE,
                )
            )

    fig.update_layout(
        title="Volcano Plot — Hit Discovery Engine (WebGL)",
        xaxis_title="Log2 Fold Change (Treatment vs Control)",
        yaxis_title="-log10(FDR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=640,
        margin=dict(l=60, r=30, t=70, b=50),
        template="plotly_white",
    )
    return fig


def build_pathway_bar(significant: pd.DataFrame) -> go.Figure:
    """Build the bar chart of significant-hit counts per pathway.

    Only annotated hits (``Pathway_Annotation != "Unassigned"``) are tallied, so
    the chart answers "which biological processes are enriched among my hits?".

    Args:
        significant: The already-filtered table of significant guides.

    Returns:
        go.Figure: A horizontal bar chart, most-enriched pathway on top, or an
        empty annotated figure when no hit carries a pathway label.
    """
    annotated = significant[significant["Pathway_Annotation"] != "Unassigned"]
    counts = annotated["Pathway_Annotation"].value_counts().sort_values()

    fig = go.Figure()
    if counts.empty:
        fig.add_annotation(
            text="No pathway-annotated hits at the current thresholds",
            showarrow=False, font=dict(size=15, color="grey"),
            x=0.5, y=0.5, xref="paper", yref="paper",
        )
    else:
        fig.add_trace(
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker_color="rgba(46, 134, 106, 0.85)",
                text=counts.values,
                textposition="outside",
            )
        )
    fig.update_layout(
        title="Significant Hits by Pathway Annotation (KEGG / GO)",
        xaxis_title="Number of significant guides",
        yaxis_title="Pathway",
        height=520,
        margin=dict(l=60, r=40, t=70, b=50),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar (global state & ingestion)
# ---------------------------------------------------------------------------
def render_sidebar() -> Tuple[pd.DataFrame, float, float, int, str, str]:
    """Render the sidebar controls and return the loaded frame + filter state.

    Owns data ingestion (upload or default) and every global filter. Returning
    the resolved state to :func:`main` keeps the tab-rendering code free of
    widget wiring.

    Returns:
        Tuple[pd.DataFrame, float, float, int, str, str]: The loaded frame, the
        FDR threshold, the absolute-LFC threshold, the base-mean count cutoff,
        the chosen NTC label, and the raw gene-search string.
    """
    st.sidebar.title("CRISPR Screen Engine")
    st.sidebar.caption("Global ingestion & filter state")

    st.sidebar.subheader("1 · Data source")
    upload = st.sidebar.file_uploader(
        "Drag & drop an sgRNA table (CSV / TSV)",
        type=["csv", "tsv", "txt"],
        help="Leave empty to load the bundled synthetic screen.",
    )
    file_bytes = upload.getvalue() if upload is not None else None
    filename = upload.name if upload is not None else None
    if upload is None:
        st.sidebar.info(f"Using default dataset: `{DEFAULT_DATA_FILE}`")

    frame = load_dataframe(file_bytes, filename, DEFAULT_DATA_FILE)

    st.sidebar.subheader("2 · Control definition")
    gene_values = sorted(frame["Target_Gene"].astype(str).unique().tolist())
    default_index = gene_values.index(DEFAULT_NTC_LABEL) if DEFAULT_NTC_LABEL in gene_values else 0
    ntc_label = st.sidebar.selectbox(
        "Non-targeting control label",
        options=gene_values,
        index=default_index,
        help="Which value in 'Target_Gene' marks your non-targeting controls.",
    )

    st.sidebar.subheader("3 · Significance filters")
    fdr_threshold = st.sidebar.select_slider(
        "FDR threshold (log scale)",
        options=list(FDR_THRESHOLD_OPTIONS),
        value=0.05,
        format_func=lambda v: f"{v:.0e}" if v < 0.01 else f"{v:g}",
        help="Guides with FDR below this value are eligible to be called hits.",
    )
    max_abs_lfc = float(np.ceil(frame["LFC_Treatment_vs_Control"].abs().max()))
    lfc_threshold = st.sidebar.slider(
        "Absolute LFC threshold",
        min_value=0.0,
        max_value=max_abs_lfc,
        value=1.0,
        step=0.1,
        help="Minimum |log2 fold change| for a guide to be called a hit.",
    )
    count_cutoff = st.sidebar.select_slider(
        "Minimum Base Mean Counts",
        options=list(COUNT_CUTOFF_OPTIONS),
        value=25,
        help="Filter out low-read guides whose statistics are unreliable.",
    )

    st.sidebar.subheader("4 · Gene search")
    search_raw = st.sidebar.text_input(
        "Highlight gene(s) on the volcano",
        value="",
        placeholder="e.g. TP53, EGFR, MYC",
        help="Comma/space separated. Matching guides glow yellow in Tab 2.",
    )

    return frame, fdr_threshold, lfc_threshold, count_cutoff, ntc_label, search_raw


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------
def render_qc_tab(frame: pd.DataFrame, categories: pd.Series, ntc_label: str) -> None:
    """Render Tab 1 — Quality Control & Library Representation.

    Args:
        frame: The loaded enrichment table.
        categories: Per-guide category labels.
        ntc_label: The chosen NTC label.
    """
    st.subheader("Quality Control & Library Representation")
    st.markdown(
        "Confirm the screen is trustworthy **before** interpreting hits: controls "
        "should sit on zero across all depths, and read depth should be a smooth "
        "log-normal with no severe dropout shoulder."
    )

    # Automated, biology-aware verdicts up top -- the "grad-student-in-a-box"
    # that reads the QC numbers so the analyst doesn't have to.
    st.markdown("#### 🧠 Automated QC assessment")
    render_qc_flags(compute_qc_flags(frame, ntc_label))
    st.divider()

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            build_ma_plot(frame, categories, ntc_label),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            build_count_histogram(frame, ntc_label),
            use_container_width=True,
        )

    ntc = frame[frame["Target_Gene"] == ntc_label]["LFC_Treatment_vs_Control"]
    c1, c2, c3 = st.columns(3)
    c1.metric("NTC median LFC", f"{ntc.median():+.3f}" if not ntc.empty else "n/a",
              help="Should be ~0 if the library is well normalised.")
    c2.metric("NTC LFC spread (SD)", f"{ntc.std():.3f}" if not ntc.empty else "n/a")
    c3.metric("Median read depth",
              f"{frame['Base_Mean_Counts'].median():,.0f}")


def render_volcano_tab(
    frame: pd.DataFrame,
    categories: pd.Series,
    search_terms: Set[str],
    search_raw: str,
) -> None:
    """Render Tab 2 — the volcano-plot hit-discovery engine.

    Args:
        frame: The loaded enrichment table.
        categories: Per-guide category labels.
        search_terms: Parsed, upper-cased search terms.
        search_raw: The raw search string (for the "not found" message).
    """
    st.subheader("Hit Discovery — The Volcano Engine")

    if search_terms:
        matched = frame["Target_Gene"].str.upper().isin(search_terms).sum()
        if matched:
            st.success(
                f"Highlighting **{matched}** guide(s) for: "
                + ", ".join(sorted(search_terms))
            )
        else:
            st.warning(f"No guides matched '{search_raw}'. Check the gene symbol.")

    st.plotly_chart(
        build_volcano_plot(frame, categories, search_terms),
        use_container_width=True,
    )
    st.caption(
        "Blue = depleted (essential / dependency) · Red = enriched (resistance / "
        "suppressor) · Grey = not significant · Faint = controls · "
        "Yellow diamonds = your search."
    )


def render_pathway_tab(
    frame: pd.DataFrame,
    categories: pd.Series,
    ntc_label: str,
) -> None:
    """Render Tab 3 — pathway enrichment, hit table and CSV export.

    Args:
        frame: The loaded enrichment table.
        categories: Per-guide category labels.
        ntc_label: The chosen NTC label (excluded from the exported hit table).
    """
    st.subheader("Pathway Enrichment & Report Export")

    is_hit = categories.isin([CAT_UP, CAT_DOWN])
    significant = frame[is_hit].copy()
    significant["Direction"] = np.where(
        significant["LFC_Treatment_vs_Control"] > 0, "Enriched", "Depleted"
    )

    st.plotly_chart(build_pathway_bar(significant), use_container_width=True)

    st.markdown("#### Filtered significant hits")
    display_cols = [
        "sgRNA_ID", "Target_Gene", "Direction", "Base_Mean_Counts",
        "LFC_Treatment_vs_Control", "p_value", "FDR", "Pathway_Annotation",
    ]
    table = significant[display_cols].sort_values("FDR").reset_index(drop=True)
    st.dataframe(
        table,
        use_container_width=True,
        height=420,
        column_config={
            "LFC_Treatment_vs_Control": st.column_config.NumberColumn(
                "LFC", format="%.3f"
            ),
            "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
            "FDR": st.column_config.NumberColumn("FDR", format="%.2e"),
            "Base_Mean_Counts": st.column_config.NumberColumn(
                "Base Mean", format="%d"
            ),
        },
    )

    # One-click export of exactly what the analyst is looking at.
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Report (filtered hits as CSV)",
        data=csv_bytes,
        file_name="crispr_filtered_significant_hits.csv",
        mime="text/csv",
        disabled=table.empty,
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Compose the full dashboard: page config, sidebar, KPI row and tabs."""
    st.set_page_config(
        page_title="CRISPR Screen Analysis Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Ingestion + global filter state (with friendly failure handling). ---
    try:
        (frame, fdr_threshold, lfc_threshold,
         count_cutoff, ntc_label, search_raw) = render_sidebar()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
        return  # Unreachable, but keeps type-checkers and readers happy.

    categories = classify_guides(
        frame, fdr_threshold, lfc_threshold, count_cutoff, ntc_label
    )
    search_terms = parse_search_terms(search_raw)

    # --- Header & KPI strip. ---
    st.title("🧬 Genome-Wide CRISPR Screen Analysis Engine")
    st.caption(
        "Interactive replacement for static enrichment files and one-off R scripts."
    )

    n_hits = int(categories.isin([CAT_UP, CAT_DOWN]).sum())
    n_up = int((categories == CAT_UP).sum())
    n_down = int((categories == CAT_DOWN).sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total guides", f"{len(frame):,}")
    k2.metric("Target genes", f"{frame['Target_Gene'].nunique() - 1:,}")
    k3.metric("Significant hits", f"{n_hits:,}")
    k4.metric("Enriched (up)", f"{n_up:,}")
    k5.metric("Depleted (down)", f"{n_down:,}")

    tab_qc, tab_volcano, tab_pathway = st.tabs(
        ["🔬 QC & Representation", "🌋 Hit Discovery", "🧭 Pathways & Export"]
    )
    with tab_qc:
        render_qc_tab(frame, categories, ntc_label)
    with tab_volcano:
        render_volcano_tab(frame, categories, search_terms, search_raw)
    with tab_pathway:
        render_pathway_tab(frame, categories, ntc_label)


if __name__ == "__main__":
    main()
