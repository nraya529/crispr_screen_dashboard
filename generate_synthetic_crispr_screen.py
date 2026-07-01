#!/usr/bin/env python3
"""Synthetic genome-wide CRISPR screen data generator.

This module fabricates a *statistically self-consistent* differential-enrichment
table that mimics the output of a pooled CRISPR knockout / Perturb-seq screen
after it has passed through a count-modelling pipeline such as **MAGeCK MLE** or
**DESeq2**. The goal is to produce a file that a wet-lab biologist would find
indistinguishable from a real ``sgRNA`` summary table, so that a downstream
dashboard can be demonstrated against believable data.

Biological framing
-------------------
A pooled screen infects a cell population with a lentiviral library where every
cell receives (ideally) a single sgRNA. Cells are split into a *control* arm and
a *treatment* arm (e.g. drug, time, or a selective pressure). Guides are counted
by amplicon sequencing at the start and end of the experiment. For each guide we
report:

* ``Base_Mean_Counts`` -- how well represented the guide is in the library
  (sequencing depth). This is the single biggest driver of statistical power.
* ``LFC_Treatment_vs_Control`` -- the log2 fold change of the guide's abundance.
  Strongly **negative** LFC == the cells carrying that knockout *dropped out*
  (the gene is *essential* / a dependency). Strongly **positive** LFC == the
  knockout was *enriched* (loss of the gene confers a growth/survival advantage,
  the classic signature of a tumour suppressor under selective pressure).
* ``p_value`` / ``FDR`` -- significance of the shift, corrected for the ~15k
  simultaneous hypotheses we are testing.

The central statistical realism requirement is the **mean--variance
relationship**: low-count guides are noisy, so even a large observed LFC there is
untrustworthy (high p-value), whereas a modest LFC on a deeply sequenced guide is
highly significant. We encode this explicitly rather than drawing the columns
independently.

Usage
-----
::

    python generate_synthetic_crispr_screen.py
    python generate_synthetic_crispr_screen.py --output my_screen.csv --seed 7

Output
------
``processed_sgRNA_enrichment.csv`` with exactly the columns required by the
companion dashboard: ``sgRNA_ID``, ``Target_Gene``, ``Base_Mean_Counts``,
``LFC_Treatment_vs_Control``, ``p_value``, ``FDR``, ``Pathway_Annotation``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Experiment-wide configuration
# ---------------------------------------------------------------------------
# These constants define the *shape* of the simulated screen. They are grouped
# here so the whole experiment can be re-tuned from a single place.

RANDOM_SEED: int = 42            # Master seed -> fully reproducible output.
N_TOTAL_GUIDES: int = 15_000     # Hard requirement: exactly 15k rows.
N_NTC_GUIDES: int = 500          # Non-targeting controls (the null distribution).
N_TARGET_GENES: int = 3_000      # Unique protein-coding targets.
N_TARGETING_GUIDES: int = N_TOTAL_GUIDES - N_NTC_GUIDES  # 14,500 -> ~4.83 g/gene.
MIN_GUIDES_PER_GENE: int = 3     # Typical modern libraries carry 4-6 guides/gene;
MAX_GUIDES_PER_GENE: int = 6     # we bracket that with a 3-6 range.

# Base-mean (library representation) model.
# Empirically, reads-per-guide in a pooled library are heavily right-skewed and
# are very well approximated by a log-normal. The per-guide count *across
# replicates* is then Negative-Binomial (a Gamma-Poisson mixture) with this
# value as its mean -- exactly the DESeq2 generative model.
LOGNORMAL_MU: float = 5.8        # exp(5.8) ~ 330 reads = median representation.
LOGNORMAL_SIGMA: float = 1.5     # Fat tail so a handful of guides reach ~50k.
COUNT_FLOOR: int = 10            # Guides below ~10 reads are usually filtered.
COUNT_CEIL: int = 50_000         # Practical upper bound on per-guide depth.

# LFC standard-error model: SE(lfc) = sqrt(floor^2 + dispersion / base_mean).
# This is the crux of the mean--variance relationship. As base_mean -> inf the SE
# collapses to SE_FLOOR (irreducible biological replicate noise); as base_mean
# -> COUNT_FLOOR the SE blows up (shot noise on a poorly sampled guide).
SE_FLOOR: float = 0.12           # Biological floor on LFC uncertainty.
COUNT_DISPERSION: float = 3.0    # Strength of the count-driven noise term.
GUIDE_EFFICIENCY_SD: float = 0.35  # Guide-to-guide spread within one gene
#                                     (different guides cut with different
#                                     efficiency, so they don't all shift equally).
NTC_EXTRA_SD: float = 0.15       # Extra technical jitter layered onto NTCs so
#                                  they show the characteristic "cloud" around 0.

# Significance thresholds used *only* to decide which hits earn a pathway label.
SIG_FDR: float = 0.05
SIG_LFC: float = 1.0

# Gene-level true-effect priors: (mean, sd) of the per-gene log2 fold change.
# "strong" == curated gold-standard genes; "mild" == randomly seeded background
# hits that keep the volcano plot from looking artificially bimodal.
MU_PARAMS: Dict[str, Tuple[float, float]] = {
    "essential_strong": (-3.2, 0.7),   # Core dependencies: ribosome, RNA Pol II.
    "essential_mild": (-1.7, 0.7),     # Context/lineage dependencies.
    "enriched_strong": (2.4, 0.6),     # Bona-fide tumour suppressors.
    "enriched_mild": (1.4, 0.6),       # Weak resistance hits.
    "neutral": (0.0, 0.10),            # The vast majority of the genome.
}

# Fraction of the *filler* genes seeded with a background effect, so the screen
# is not dominated purely by the hand-curated hallmark genes.
FRAC_FILLER_ESSENTIAL: float = 0.08
FRAC_FILLER_ENRICHED: float = 0.03
FRAC_SIG_HITS_WITH_PATHWAY: float = 0.30  # ~30% of hits get a KEGG/GO label.

OUTPUT_FILENAME: str = "processed_sgRNA_enrichment.csv"

# ---------------------------------------------------------------------------
# Curated hallmark gene sets (real HGNC symbols with real screen phenotypes)
# ---------------------------------------------------------------------------
# Ribosomal proteins are the canonical "gold-standard essential" set (Hart et al.
# 2015): knocking any of them out is lethal in essentially every cell line, so
# they anchor the strongly-depleted end of the distribution.
RIBOSOMAL_GENES: List[str] = (
    [f"RPL{i}" for i in range(1, 41)] + [f"RPS{i}" for i in range(1, 31)]
)

# RNA Polymerase II subunits + spliceosome + proteasome + DNA-replication +
# mitosis machinery -- all pan-essential housekeeping complexes.
CORE_ESSENTIAL_GENES: List[str] = [
    "MYC", "POLR2A", "POLR2B", "POLR2C", "POLR2D", "POLR2E", "POLR2F",
    "POLR2G", "POLR2H", "POLR2I", "POLR2L",
    "EIF4A3", "EIF4G1", "EIF3B", "SF3B1", "SF3B3", "U2AF1", "SNRPD1",
    "SNRPD2", "PRPF8", "PRPF19",
    "CDK1", "CDK9", "CDK11B", "PLK1", "AURKB", "BUB1B", "BUB3", "KIF11",
    "INCENP", "NDC80", "CENPA",
    "RRM1", "RRM2", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5",
    "MCM6", "MCM7", "POLA1", "POLD1", "POLE",
    "RAN", "NUP93", "PSMA1", "PSMB2", "PSMC1", "PSMD1", "CCT2", "TCP1",
]

# Oxidative-phosphorylation / mitochondrial complex members -- essential in
# lines that depend on respiration; also used to seed the OXPHOS pathway label.
OXPHOS_GENES: List[str] = [
    "NDUFA1", "NDUFA2", "NDUFB4", "NDUFS1", "SDHA", "SDHB", "UQCRC1",
    "UQCRC2", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "ATP5MC1",
]

# Tumour suppressors: loss confers a fitness *advantage* under selective
# pressure, hence a positive (enriched) LFC.
TUMOR_SUPPRESSOR_GENES: List[str] = [
    "TP53", "PTEN", "RB1", "NF1", "NF2", "CDKN2A", "CDKN1A", "APC", "VHL",
    "STK11", "KEAP1", "SMAD4", "BAP1", "ARID1A", "PBRM1", "SETD2", "FBXW7",
    "MEN1", "TSC1", "TSC2", "MLH1", "MSH2",
]

# Kinases / oncogenes -- mostly neutral in a generic screen, but a subset drive
# "oncogene addiction" and therefore deplete. Kept as searchable landmarks.
KINASE_ONCOGENE_GENES: List[str] = [
    "EGFR", "ERBB2", "ERBB3", "MET", "ALK", "BRAF", "RAF1", "KRAS", "NRAS",
    "HRAS", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "PIK3CA", "PIK3CB",
    "AKT1", "AKT2", "MTOR", "SRC", "JAK1", "JAK2", "ABL1", "KIT", "PDGFRA",
    "FGFR1", "FGFR2", "CDK4", "CDK6", "CCND1", "MDM2", "BCL2", "MCL1",
    "BAX", "BAK1", "CASP3", "CASP9", "APAF1",
]

# The subset of the above that behaves as an essential dependency.
ONCOGENE_ADDICTION_GENES: List[str] = [
    "EGFR", "BRAF", "KRAS", "MTOR", "CDK4", "CDK6", "MCL1", "BCL2",
    "PIK3CA", "MET", "CCND1",
]

# Realistic prefixes for procedurally-generated "background" gene symbols. These
# are all real, very large HGNC gene families, so the filler names look native.
FILLER_PREFIXES: List[str] = [
    "ZNF", "SLC", "TMEM", "FAM", "CCDC", "GPR", "OR", "KRT", "CYP", "ABCA",
    "ABCB", "ABCC", "COL", "MMP", "ADAM", "USP", "KDM", "OLFM", "PCDH",
    "DNAJ", "RAB", "ARHGAP", "ANKRD", "WDR", "TRIM",
]

# Universe of KEGG/GO pathway terms used for annotation.
PATHWAYS: List[str] = [
    "p53 signaling pathway",
    "Oxidative phosphorylation",
    "Cell cycle",
    "Ribosome",
    "MAPK signaling pathway",
    "PI3K-Akt signaling pathway",
    "DNA replication",
    "Apoptosis",
    "mRNA surveillance pathway",
    "RNA polymerase",
    "Spliceosome",
    "Proteasome",
]

NTC_LABEL: str = "NTC"
UNASSIGNED_LABEL: str = "Unassigned"


# ---------------------------------------------------------------------------
# Gene catalogue construction
# ---------------------------------------------------------------------------
def _build_pathway_map() -> Dict[str, str]:
    """Map curated gene symbols to their canonical KEGG/GO pathway term.

    The mapping is a mix of explicit assignments (for the hand-curated hallmark
    genes) and prefix rules (for whole families such as ribosomal proteins). It
    encodes real membership so that, when a curated gene turns out to be a
    significant hit, it is annotated with a biologically correct term rather than
    a random one.

    Returns:
        Dict[str, str]: Gene symbol -> pathway term for every gene we can place
        with confidence. Genes absent from the map are considered unannotated.
    """
    mapping: Dict[str, str] = {}

    # Family / prefix rules ------------------------------------------------
    for gene in RIBOSOMAL_GENES:
        mapping[gene] = "Ribosome"
    for gene in OXPHOS_GENES:
        mapping[gene] = "Oxidative phosphorylation"
    for gene in CORE_ESSENTIAL_GENES:
        if gene.startswith("POLR2"):
            mapping[gene] = "RNA polymerase"
        elif gene.startswith("MCM") or gene.startswith("POL") or gene in {"PCNA", "RRM1", "RRM2"}:
            mapping[gene] = "DNA replication"
        elif gene.startswith("PSM"):
            mapping[gene] = "Proteasome"
        elif gene in {"SF3B1", "SF3B3", "U2AF1", "SNRPD1", "SNRPD2", "PRPF8", "PRPF19", "EIF4A3"}:
            mapping[gene] = "Spliceosome"
        else:
            mapping[gene] = "Cell cycle"

    # Explicit signalling / apoptosis assignments -------------------------
    p53_axis = {"TP53", "MDM2", "CDKN1A", "CDKN2A", "RB1", "BAX", "BAK1"}
    mapk_axis = {"EGFR", "ERBB2", "ERBB3", "MET", "BRAF", "RAF1", "KRAS",
                 "NRAS", "HRAS", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
                 "FGFR1", "FGFR2", "KIT", "PDGFRA"}
    pi3k_axis = {"PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR", "PTEN",
                 "TSC1", "TSC2", "STK11"}
    apoptosis_axis = {"BCL2", "MCL1", "CASP3", "CASP9", "APAF1"}
    cellcycle_axis = {"CDK4", "CDK6", "CCND1"}

    for gene in p53_axis:
        mapping.setdefault(gene, "p53 signaling pathway")
    for gene in mapk_axis:
        mapping.setdefault(gene, "MAPK signaling pathway")
    for gene in pi3k_axis:
        mapping.setdefault(gene, "PI3K-Akt signaling pathway")
    for gene in apoptosis_axis:
        mapping.setdefault(gene, "Apoptosis")
    for gene in cellcycle_axis:
        mapping.setdefault(gene, "Cell cycle")

    return mapping


def _generate_filler_genes(existing: List[str], n_needed: int,
                           rng: np.random.Generator) -> List[str]:
    """Procedurally synthesise realistic, unique background gene symbols.

    We draw ``PREFIX + integer`` combinations from real, large gene families so
    that the ~2,800 background genes look like genuine HGNC symbols rather than
    ``GENE0001`` placeholders.

    Args:
        existing: Symbols already in use (curated genes) that must not collide.
        n_needed: How many additional unique symbols to produce.
        rng: Seeded NumPy generator for reproducibility.

    Returns:
        List[str]: ``n_needed`` unique gene symbols, none of which appear in
        ``existing``.
    """
    used = set(existing)
    filler: List[str] = []
    while len(filler) < n_needed:
        prefix = rng.choice(FILLER_PREFIXES)
        number = int(rng.integers(1, 900))
        symbol = f"{prefix}{number}"
        if symbol not in used:
            used.add(symbol)
            filler.append(symbol)
    return filler


def _assign_guides_per_gene(rng: np.random.Generator) -> np.ndarray:
    """Distribute exactly ``N_TARGETING_GUIDES`` guides across the target genes.

    Every gene is guaranteed a floor of ``MIN_GUIDES_PER_GENE`` guides; the
    remaining budget is scattered so that no gene exceeds ``MAX_GUIDES_PER_GENE``.
    This reproduces the "3-6 guides per gene" design of a modern library while
    hitting the total row count exactly.

    Args:
        rng: Seeded NumPy generator.

    Returns:
        np.ndarray: Integer array of length ``N_TARGET_GENES`` that sums to
        ``N_TARGETING_GUIDES``.
    """
    base = np.full(N_TARGET_GENES, MIN_GUIDES_PER_GENE, dtype=int)
    remaining = N_TARGETING_GUIDES - base.sum()
    # Each gene can absorb up to (MAX - MIN) extra guides. Build one "slot" per
    # unit of spare capacity, shuffle, and fill the first `remaining` of them.
    spare_slots = np.repeat(
        np.arange(N_TARGET_GENES), MAX_GUIDES_PER_GENE - MIN_GUIDES_PER_GENE
    )
    rng.shuffle(spare_slots)
    chosen = spare_slots[:remaining]
    extra = np.bincount(chosen, minlength=N_TARGET_GENES)
    return base + extra


def build_gene_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Assemble the 3,000-gene catalogue with categories and true effect sizes.

    Curated hallmark genes are locked into their known biological category;
    background genes are randomly seeded with mild effects at the configured
    frequencies. Every gene is then assigned a single latent ``gene_mu`` -- the
    true per-gene log2 fold change shared by all of its guides.

    Args:
        rng: Seeded NumPy generator.

    Returns:
        pd.DataFrame: One row per gene with columns ``Target_Gene``,
        ``category``, ``gene_mu``.
    """
    # De-duplicate the curated sets while preserving priority ordering.
    curated: List[str] = []
    for group in (RIBOSOMAL_GENES, CORE_ESSENTIAL_GENES, OXPHOS_GENES,
                  TUMOR_SUPPRESSOR_GENES, KINASE_ONCOGENE_GENES):
        for gene in group:
            if gene not in curated:
                curated.append(gene)

    n_filler = N_TARGET_GENES - len(curated)
    if n_filler < 0:
        raise ValueError(
            f"Curated gene set ({len(curated)}) exceeds N_TARGET_GENES "
            f"({N_TARGET_GENES}). Trim the curated lists or raise N_TARGET_GENES."
        )
    filler = _generate_filler_genes(curated, n_filler, rng)
    all_genes = curated + filler

    strong_essential = set(RIBOSOMAL_GENES) | set(CORE_ESSENTIAL_GENES) | set(OXPHOS_GENES)
    mild_essential = set(ONCOGENE_ADDICTION_GENES)
    strong_enriched = set(TUMOR_SUPPRESSOR_GENES)

    categories: List[str] = []
    for gene in all_genes:
        if gene in strong_essential:
            categories.append("essential_strong")
        elif gene in strong_enriched:
            categories.append("enriched_strong")
        elif gene in mild_essential:
            categories.append("essential_mild")
        elif gene in set(KINASE_ONCOGENE_GENES):
            # Remaining kinases/oncogenes are neutral landmarks (searchable but
            # usually not significant in a generic screen).
            categories.append("neutral")
        else:
            # Background gene: seed a minority with mild effects.
            draw = rng.random()
            if draw < FRAC_FILLER_ESSENTIAL:
                categories.append("essential_mild")
            elif draw < FRAC_FILLER_ESSENTIAL + FRAC_FILLER_ENRICHED:
                categories.append("enriched_mild")
            else:
                categories.append("neutral")

    # Draw the latent per-gene effect from its category prior.
    means = np.array([MU_PARAMS[c][0] for c in categories])
    sds = np.array([MU_PARAMS[c][1] for c in categories])
    gene_mu = rng.normal(loc=means, scale=sds)

    return pd.DataFrame(
        {"Target_Gene": all_genes, "category": categories, "gene_mu": gene_mu}
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Implemented directly (no statsmodels dependency) so the statistical core is
    transparent and auditable. The step-up procedure controls the expected
    proportion of false discoveries across the ~15k simultaneous guide-level
    tests -- without it, ~5% of the genome (750 guides) would appear "significant"
    by chance alone.

    Args:
        p_values: 1-D array of raw p-values in ``[0, 1]``.

    Returns:
        np.ndarray: Adjusted p-values (q-values), aligned to the input order and
        clipped to ``[0, 1]``.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)                     # Ascending p-value order.
    ranked = p[order]
    ranks = np.arange(1, n + 1)
    # q_i = p_i * n / rank_i, then enforce monotonicity from the largest p down
    # so that q-values are non-decreasing in p (the standard BH "step-up" fix).
    q = ranked * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def simulate_screen_statistics(
    base_mean: np.ndarray,
    guide_true_lfc: np.ndarray,
    extra_sd: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate observed LFC, raw p-values and the standard error per guide.

    This is where the mean--variance relationship is enforced. Each guide's LFC
    standard error is::

        SE = sqrt(SE_FLOOR^2 + COUNT_DISPERSION / base_mean + extra_sd^2)

    A Wald test statistic ``z = observed_LFC / SE`` is then converted to a
    two-sided p-value. The consequence is exactly the behaviour a bioinformatician
    expects:

    * Deeply-sequenced guide + large true LFC  -> tiny SE, huge |z|, tiny p.
    * Poorly-sequenced guide (low base_mean)   -> large SE, deflated |z|, large p
      *even if its point-estimate LFC looks large* -- the shot-noise punishes it.

    Args:
        base_mean: Per-guide mean normalised counts (library representation).
        guide_true_lfc: The latent true log2 fold change for each guide.
        extra_sd: Additional per-guide noise (used to inflate NTC variance);
            pass an array of zeros for ordinary targeting guides.
        rng: Seeded NumPy generator.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(observed_lfc, p_value, se)``.
    """
    se = np.sqrt(SE_FLOOR ** 2 + COUNT_DISPERSION / base_mean + extra_sd ** 2)
    # Observed estimate = truth + measurement noise scaled by the guide's SE.
    observed_lfc = guide_true_lfc + rng.normal(loc=0.0, scale=se)
    z = observed_lfc / se
    # Two-sided Wald p-value; floor away from exactly 0 to avoid -inf downstream.
    p_value = 2.0 * stats.norm.sf(np.abs(z))
    p_value = np.clip(p_value, 1e-300, 1.0)
    return observed_lfc, p_value, se


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _draw_base_means(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw right-skewed per-guide library representation (mean counts).

    Reads-per-guide in a pooled library are heavily right-skewed; a log-normal
    reproduces that shape well. The returned value is the *base mean* -- the mean
    of the underlying Negative-Binomial (Gamma-Poisson) count distribution -- and
    is rounded to an integer count and clipped to a realistic depth window.

    Args:
        n: Number of guides.
        rng: Seeded NumPy generator.

    Returns:
        np.ndarray: Integer base-mean counts in ``[COUNT_FLOOR, COUNT_CEIL]``.
    """
    raw = rng.lognormal(mean=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA, size=n)
    clipped = np.clip(raw, COUNT_FLOOR, COUNT_CEIL)
    return np.round(clipped).astype(int)


def build_targeting_block(catalog: pd.DataFrame,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Expand the gene catalogue into individual targeting-sgRNA rows.

    Args:
        catalog: Output of :func:`build_gene_catalog`.
        rng: Seeded NumPy generator.

    Returns:
        pd.DataFrame: One row per targeting guide with the observed columns plus
        a working ``category`` column (dropped before writing the CSV).
    """
    guides_per_gene = _assign_guides_per_gene(rng)

    genes = catalog["Target_Gene"].to_numpy()
    gene_mu = catalog["gene_mu"].to_numpy()
    categories = catalog["category"].to_numpy()

    # Vector-expand every gene-level attribute to guide resolution.
    gene_repeat = np.repeat(genes, guides_per_gene)
    mu_repeat = np.repeat(gene_mu, guides_per_gene)
    cat_repeat = np.repeat(categories, guides_per_gene)

    # Per-guide identifier: <GENE>_sg<k>, k running 1..guides_for_that_gene.
    guide_index = np.concatenate([np.arange(1, g + 1) for g in guides_per_gene])
    sgrna_ids = [f"{gene}_sg{k}" for gene, k in zip(gene_repeat, guide_index)]

    n = gene_repeat.size
    base_mean = _draw_base_means(n, rng)
    # Guides of one gene do not shift identically: guide efficiency varies, which
    # we model as Gaussian jitter around the gene's latent effect.
    guide_true_lfc = mu_repeat + rng.normal(0.0, GUIDE_EFFICIENCY_SD, size=n)
    observed_lfc, p_value, _ = simulate_screen_statistics(
        base_mean, guide_true_lfc, extra_sd=np.zeros(n), rng=rng
    )

    return pd.DataFrame(
        {
            "sgRNA_ID": sgrna_ids,
            "Target_Gene": gene_repeat,
            "Base_Mean_Counts": base_mean,
            "LFC_Treatment_vs_Control": observed_lfc,
            "p_value": p_value,
            "category": cat_repeat,
        }
    )


def build_ntc_block(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the non-targeting control (NTC) guides.

    NTCs cut nowhere in the genome, so their *true* LFC is exactly 0 -- they exist
    to measure the technical noise floor and to verify library normalisation. We
    therefore centre them at 0 but inflate their variance (via ``NTC_EXTRA_SD``)
    so they form the characteristic diffuse cloud around LFC = 0 on the MA plot.
    Because they are drawn from the null, a handful will cross significance by
    chance -- a realistic and useful sanity signal for the analyst.

    Args:
        rng: Seeded NumPy generator.

    Returns:
        pd.DataFrame: ``N_NTC_GUIDES`` rows sharing the targeting-block schema.
    """
    base_mean = _draw_base_means(N_NTC_GUIDES, rng)
    # True effect is ~0 (controls target nothing); keep a whisper of spread.
    guide_true_lfc = rng.normal(0.0, 0.05, size=N_NTC_GUIDES)
    observed_lfc, p_value, _ = simulate_screen_statistics(
        base_mean,
        guide_true_lfc,
        extra_sd=np.full(N_NTC_GUIDES, NTC_EXTRA_SD),
        rng=rng,
    )
    sgrna_ids = [f"{NTC_LABEL}_{i:04d}" for i in range(1, N_NTC_GUIDES + 1)]

    return pd.DataFrame(
        {
            "sgRNA_ID": sgrna_ids,
            "Target_Gene": NTC_LABEL,
            "Base_Mean_Counts": base_mean,
            "LFC_Treatment_vs_Control": observed_lfc,
            "p_value": p_value,
            "category": "ntc",
        }
    )


def assign_pathways(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Annotate ~30% of significant hits with a KEGG/GO pathway term.

    A guide is eligible if it clears both the FDR and |LFC| thresholds and is not
    an NTC. Of the eligible hits, a random ``FRAC_SIG_HITS_WITH_PATHWAY`` fraction
    receive a term: the biologically correct one when the gene is in the curated
    pathway map, otherwise a randomly drawn term (representing the reality that
    many hits map to *some* annotated process). Everything else is ``Unassigned``.

    Args:
        df: The full guide table (must contain ``FDR`` by this point).
        rng: Seeded NumPy generator.

    Returns:
        pd.Series: The ``Pathway_Annotation`` column aligned to ``df``.
    """
    pathway_map = _build_pathway_map()
    annotation = np.full(len(df), UNASSIGNED_LABEL, dtype=object)

    is_significant = (
        (df["FDR"].to_numpy() < SIG_FDR)
        & (df["LFC_Treatment_vs_Control"].abs().to_numpy() >= SIG_LFC)
        & (df["Target_Gene"].to_numpy() != NTC_LABEL)
    )
    sig_positions = np.where(is_significant)[0]
    n_to_label = int(round(FRAC_SIG_HITS_WITH_PATHWAY * sig_positions.size))
    if n_to_label > 0:
        chosen = rng.choice(sig_positions, size=n_to_label, replace=False)
        genes = df["Target_Gene"].to_numpy()
        for pos in chosen:
            gene = genes[pos]
            annotation[pos] = pathway_map.get(gene, rng.choice(PATHWAYS))

    return pd.Series(annotation, index=df.index, name="Pathway_Annotation")


def generate_screen(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Run the full simulation and return the final, ordered guide table.

    Args:
        seed: Master random seed controlling every stochastic draw.

    Returns:
        pd.DataFrame: Exactly ``N_TOTAL_GUIDES`` rows with the seven required
        output columns, in analysis-ready order.
    """
    rng = np.random.default_rng(seed)

    catalog = build_gene_catalog(rng)
    targeting = build_targeting_block(catalog, rng)
    ntc = build_ntc_block(rng)

    # Concatenate, then compute FDR *once* across the whole library -- multiple
    # testing must be corrected over every hypothesis in the experiment jointly.
    df = pd.concat([targeting, ntc], ignore_index=True)
    df["FDR"] = benjamini_hochberg(df["p_value"].to_numpy())
    df["Pathway_Annotation"] = assign_pathways(df, rng)

    # Final column selection & tidy rounding for a clean, human-readable CSV.
    df["Base_Mean_Counts"] = df["Base_Mean_Counts"].astype(int)
    df["LFC_Treatment_vs_Control"] = df["LFC_Treatment_vs_Control"].round(4)
    df["p_value"] = df["p_value"]
    df["FDR"] = df["FDR"]

    ordered = df[
        [
            "sgRNA_ID",
            "Target_Gene",
            "Base_Mean_Counts",
            "LFC_Treatment_vs_Control",
            "p_value",
            "FDR",
            "Pathway_Annotation",
        ]
    ].copy()
    return ordered


def _print_summary(df: pd.DataFrame) -> None:
    """Emit a concise QC summary of the generated screen to stdout.

    Args:
        df: The final guide table returned by :func:`generate_screen`.
    """
    is_sig = (df["FDR"] < SIG_FDR) & (df["LFC_Treatment_vs_Control"].abs() >= SIG_LFC)
    targeting = df[df["Target_Gene"] != NTC_LABEL]
    ntc = df[df["Target_Gene"] == NTC_LABEL]
    depleted = df[is_sig & (df["LFC_Treatment_vs_Control"] < 0)]
    enriched = df[is_sig & (df["LFC_Treatment_vs_Control"] > 0)]

    print("=" * 64)
    print("SYNTHETIC CRISPR SCREEN -- GENERATION SUMMARY")
    print("=" * 64)
    print(f"Total guides ............ {len(df):>8,}")
    print(f"  Targeting guides ...... {len(targeting):>8,}")
    print(f"  Unique target genes ... {targeting['Target_Gene'].nunique():>8,}")
    print(f"  NTC guides ............ {len(ntc):>8,}")
    print(f"Significant hits ........ {int(is_sig.sum()):>8,} "
          f"(FDR<{SIG_FDR}, |LFC|>={SIG_LFC})")
    print(f"  Depleted (essential) .. {len(depleted):>8,}")
    print(f"  Enriched (suppressor) . {len(enriched):>8,}")
    print(f"Base-mean range ......... "
          f"{df['Base_Mean_Counts'].min():,} - {df['Base_Mean_Counts'].max():,}")
    print(f"NTC mean LFC ............ {ntc['LFC_Treatment_vs_Control'].mean():+.4f} "
          f"(sd {ntc['LFC_Treatment_vs_Control'].std():.4f})")
    n_annotated = int((df['Pathway_Annotation'] != UNASSIGNED_LABEL).sum())
    print(f"Pathway-annotated hits .. {n_annotated:>8,}")
    print("Top depleted genes ...... "
          + ", ".join(
              depleted.groupby("Target_Gene")["LFC_Treatment_vs_Control"]
              .mean().nsmallest(8).index.tolist()
          ))
    print("=" * 64)


def main() -> None:
    """Command-line entry point: generate the screen and write the CSV."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic genome-wide CRISPR screen enrichment table."
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_FILENAME,
        help=f"Output CSV path (default: {OUTPUT_FILENAME}).",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    args = parser.parse_args()

    df = generate_screen(seed=args.seed)
    df.to_csv(args.output, index=False)
    _print_summary(df)
    print(f"\nWrote {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
