# 🧬 CRISPR Screen Analysis Engine

An interactive web app for exploring genome-wide CRISPR screen results
(Perturb-seq / pooled knockout). It replaces static enrichment files and one-off
R scripts with live filtering, WebGL-accelerated plots, automated QC verdicts,
and one-click export.

- **QC & Representation** — MA plot, read-depth distribution, and an automated,
  biology-aware QC panel (control spread/centring, dropout, and a pan-essential
  positive-control "did the screen work?" check).
- **Hit Discovery** — a WebGL volcano plot over ~15k guides with adjustable
  FDR / |LFC| / read-depth thresholds and a gene search that highlights guides.
- **Pathways & Export** — pathway-enrichment bar chart, an interactive hit table,
  and a Download Report button.

## Quick start (local)

```bash
cd crispr_screen_dashboard
python3 -m pip install -r requirements.txt

# 1. Generate the synthetic demo dataset (creates processed_sgRNA_enrichment.csv)
python3 generate_synthetic_crispr_screen.py

# 2. Launch the dashboard (opens http://localhost:8501)
python3 -m streamlit run crispr_dashboard_engine.py
```

## Uploading your own data

Drag any CSV/TSV into the sidebar uploader. Headers from **MAGeCK**, **DESeq2**,
and **edgeR/limma** are auto-detected and mapped, so raw exports usually work
with no renaming. The canonical columns the app needs are:

| Canonical column          | Auto-detected aliases (examples)                    |
|---------------------------|-----------------------------------------------------|
| `sgRNA_ID`                | `sgrna`, `guide`, `barcode`, `id`                   |
| `Target_Gene`             | `Gene`, `gene_symbol`, `symbol`                     |
| `Base_Mean_Counts`        | `baseMean`, `mean_counts` (or derived from counts)  |
| `LFC_Treatment_vs_Control`| `LFC`, `log2FoldChange`, `logFC`                    |
| `p_value`                 | `pvalue`, `p.twosided`, `pval`                      |
| `FDR`                     | `padj`, `fdr`, `q_value`                             |
| `Pathway_Annotation`      | optional — filled as `Unassigned` when absent       |

If a required column can't be matched, the app tells you exactly which columns it
found vs. expected. To teach it a new header, add it to `COLUMN_ALIASES` in
`crispr_dashboard_engine.py`.

## Deploy to the web (Streamlit Community Cloud, free)

Anyone with the resulting URL can use it in a browser — no local install.

1. **Push this folder to a public GitHub repo:**
   ```bash
   cd crispr_screen_dashboard
   git init
   git add .
   git commit -m "CRISPR screen dashboard"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/crispr-dashboard.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io**, sign in with GitHub, click **New app**.
3. Set **Repository** = `<YOUR_USERNAME>/crispr-dashboard`, **Branch** = `main`,
   **Main file path** = `crispr_dashboard_engine.py`, then **Deploy**.

Streamlit installs `requirements.txt` automatically and serves the app in a
couple of minutes. The committed `processed_sgRNA_enrichment.csv` is the default
dataset shown before any upload.

## Files

| File                                   | Purpose                                    |
|----------------------------------------|--------------------------------------------|
| `generate_synthetic_crispr_screen.py`  | Statistically realistic demo-data generator |
| `crispr_dashboard_engine.py`           | The Streamlit + Plotly application          |
| `requirements.txt`                     | Pinned dependencies                         |
| `.streamlit/config.toml`               | Theme + upload limits + no first-run prompt |
| `processed_sgRNA_enrichment.csv`       | Default dataset (generated)                 |
