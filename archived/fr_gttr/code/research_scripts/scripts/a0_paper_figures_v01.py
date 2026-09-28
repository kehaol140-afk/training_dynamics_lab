#!/usr/bin/env python3
"""Render paper-style figures from the frozen FR_A0C/A0D/A0E/A0F/A0F2 artifacts.

[FR-OBSERVATION-ONLY] This script reads frozen summary.json artifacts and
renders publication figures; it never regenerates observations and never
mutates any experiment output. Every figure is written as vector PDF plus a
600-dpi PNG companion into outputs/a0_fr_audit/paper_figures_v01/.

Figure inventory
----------------
fig1_a0d_epsilon_convergence : A0D epsilon-halving trajectories of the
    stream-mean finite difference chi_FD(eps) toward the frozen full-tangent
    value chi_tangent, with the O(eps^2) slope guide on the error panel.
fig2_a0f2_parity             : A0F2 census parity plots (prediction vs the
    registered finite-difference response) for optimizer-only B and full
    tangent C across all 56 states and horizons H=1,2,4,8.
fig3_a0f2_error_ladder       : A0F2 per-horizon mean relative error (std
    bands) of models A/B/C over the 56-state census.
fig4_a0e_decomposition       : A0E contribution decomposition
    A -> B -> BH -> C on the mean relative error over the 3-state ladder.
fig5_a0f2_improvement        : A0F2 paired per-state B-to-C improvement
    distribution for H=2,4,8 over the 56-state census.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = PROJECT_ROOT / "outputs/a0_fr_audit/paper_figures_v01"

A0D_SUMMARY = PROJECT_ROOT / "outputs/a0_fr_audit/response_propagation_v02_epsilon/summary.json"
A0E_SUMMARY = PROJECT_ROOT / "outputs/a0_fr_audit/curvature_decomposition_v01/summary.json"
A0F2_SUMMARY = PROJECT_ROOT / "outputs/a0_fr_audit/cross_partition_extension_v01/summary.json"

# Colorblind-safe palette (Okabe-Ito).
C_A = "#0072B2"   # blue
C_B = "#E69F00"   # orange
C_BH = "#999999"  # gray
C_C = "#009E73"   # green
H_COLORS = {1: "#0072B2", 2: "#E69F00", 4: "#009E73", 8: "#D55E00"}
H_MARKERS = {1: "o", 2: "s", 4: "^", 8: "D"}
HORIZONS = (1, 2, 4, 8)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise ValueError(f"artifact is not complete: {path}")
    return payload


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"FIGURE {name}.pdf/.png")


# ---------------------------------------------------------------------------
# fig1: A0D epsilon convergence trajectories (2 rows x 3 state columns).
# ---------------------------------------------------------------------------
def fig1_a0d_epsilon_convergence(a0d: dict) -> None:
    cells = {}
    for row in a0d["convergence_results"]:
        cells[(int(row["seed"]), int(row["step"]), int(row["H"]))] = row
    state_order = [
        (int(s["seed"]), int(s["step"])) for s in a0d["scope"]["states"]
    ]
    hs = [int(h) for h in a0d["scope"]["horizons"]]

    fig, axes = plt.subplots(
        2, 3, figsize=(7.0, 4.4), sharex="col",
        gridspec_kw={"hspace": 0.32, "wspace": 0.28},
    )
    handles = {}
    for col, (seed, step) in enumerate(state_order):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        eps_reg = None
        for h in hs:
            cell = cells[(seed, step, h)]
            eps = np.asarray(cell["epsilon_descending"], dtype=float)
            chi = np.asarray(cell["chi_fd_descending"], dtype=float)
            rel = np.asarray(cell["relative_error_descending"], dtype=float)
            chi_t = float(cell["chi_tangent"])
            eps_reg = float(eps[0])
            color = H_COLORS[h]
            # Top: FD trajectory approaching the frozen tangent value.
            (line,) = ax_top.plot(
                eps, chi, marker=H_MARKERS[h], color=color,
                markerfacecolor="white", markeredgewidth=1.0,
                label=rf"$H={h}$",
            )
            handles[h] = line
            ax_top.axhline(chi_t, color=color, linestyle="--", linewidth=0.9, alpha=0.85)
            # Bottom: relative error trajectory with O(eps^2) guide.
            ax_bot.plot(
                eps, rel, marker=H_MARKERS[h], color=color,
                markerfacecolor="white", markeredgewidth=1.0,
            )
            guide = rel[0] * (eps / eps[0]) ** 2
            ax_bot.plot(eps, guide, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
        ax_top.set_xscale("log")
        ax_bot.set_xscale("log")
        ax_bot.set_yscale("log")
        ax_top.set_title(
            f"seed {seed}, step {step}\n"
            rf"$\epsilon_{{\mathrm{{reg}}}}={eps_reg:.3g}$",
            pad=3, fontsize=8, loc="right",
        )
        ax_top.grid(True, which="major")
        ax_bot.grid(True, which="major")
        ax_bot.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # Direction marker: epsilon decreases to the left.
        ax_bot.text(
            0.03, 0.90, r"$\epsilon \to 0$",
            transform=ax_bot.transAxes, fontsize=7.5, color="0.25", va="top",
        )
        ax_bot.text(
            0.03, 0.06,
            r"solid: data; dotted: $O(\epsilon^{2})$ guide",
            transform=ax_bot.transAxes, fontsize=6.8, color="0.25",
        )
        ax_bot.set_xlabel(r"perturbation scale $\epsilon$")
        if col == 0:
            ax_top.set_ylabel(r"response $\chi$")
            ax_bot.set_ylabel(r"rel. error of $\chi_{\mathrm{FD}}$")
    fig.legend(
        [handles[h] for h in hs], [rf"$H={h}$" for h in hs],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.015),
        handlelength=1.8, columnspacing=1.6,
    )
    for tag, ax in zip("abcdef", [a for row in axes for a in row]):
        ax.text(
            -0.16, 1.04, f"({tag})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left",
        )
    fig.align_ylabels(axes)
    _save(fig, "fig1_a0d_epsilon_convergence")


# ---------------------------------------------------------------------------
# fig2: A0F2 parity plots for B and C.
# ---------------------------------------------------------------------------
def fig2_a0f2_parity(a0f2: dict) -> None:
    rows = a0f2["results"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    panels = [
        ("B_optimizer_only", r"optimizer-only tangent $B$", C_B, "chi_optimizer_only"),
        ("C_full_tangent", r"full tangent $C$", C_C, "chi_full_tangent"),
    ]
    all_x = [float(r["chi_fd"]) for r in rows]
    all_y = [
        float(r["chi_optimizer_only"]) for r in rows
    ] + [float(r["chi_full_tangent"]) for r in rows]
    lo = min(min(all_x), min(all_y))
    hi = max(max(all_x), max(all_y))
    span = hi - lo
    lo -= 0.04 * span
    hi += 0.04 * span
    for ax, (label, title, color_unused, field) in zip(axes, panels):
        for h in HORIZONS:
            group = [r for r in rows if int(r["H"]) == h]
            x = np.asarray([float(r["chi_fd"]) for r in group])
            y = np.asarray([float(r[field]) for r in group])
            ax.plot(
                x, y, linestyle="none", marker=H_MARKERS[h],
                color=H_COLORS[h], markerfacecolor="white",
                markeredgewidth=0.9, markersize=4.5, alpha=0.85,
                label=rf"$H={h}$",
            )
        ax.plot([lo, hi], [lo, hi], color="0.2", linestyle="--", linewidth=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        overall = a0f2["metrics"][label]["overall"]
        ax.text(
            0.04, 0.96,
            rf"$R^{{2}}={overall['R2']:.4f}$" + "\n"
            + rf"mean rel. err. $={overall['mean_relative_error']:.4g}$",
            transform=ax.transAxes, va="top", fontsize=7.5, color="0.15",
        )
        ax.set_title(title, pad=3)
        ax.grid(True, which="major")
        ax.set_xlabel(r"observed $\chi_{\mathrm{FD}}$ (56-state census)")
    axes[0].set_ylabel(r"predicted $\chi$")
    axes[1].legend(
        loc="lower right", ncol=2, columnspacing=0.9, handletextpad=0.4,
        borderaxespad=0.2,
    )
    for tag, ax in zip("ab", axes):
        ax.text(
            -0.18, 1.04, f"({tag})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left",
        )
    _save(fig, "fig2_a0f2_parity")


# ---------------------------------------------------------------------------
# fig3: A0F2 per-horizon mean relative error ladder for A/B/C.
# ---------------------------------------------------------------------------
def fig3_a0f2_error_ladder(a0f2: dict) -> None:
    dist = a0f2["error_distributions"]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    models = [
        ("A_momentum", r"momentum-only $A$", C_A, "o"),
        ("B_optimizer_only", r"optimizer-only $B$", C_B, "s"),
        ("C_full_tangent", r"full tangent $C$", C_C, "D"),
    ]
    for label, title, color, marker in models:
        means = [
            dist[label][str(h)]["relative_error"]["mean"] for h in HORIZONS
        ]
        stds = [
            dist[label][str(h)]["relative_error"]["std"] for h in HORIZONS
        ]
        ax.errorbar(
            HORIZONS, means, yerr=stds, marker=marker, color=color,
            markerfacecolor="white", markeredgewidth=1.0, capsize=2.5,
            elinewidth=0.9, label=title, linestyle="-",
        )
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 30.0)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel(r"prediction horizon $H$ (optimizer steps)")
    ax.set_ylabel(r"mean relative error $\pm$ std")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", handlelength=1.6)
    ax.text(
        -0.20, 1.04, "(a)", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="left",
    )
    _save(fig, "fig3_a0f2_error_ladder")


# ---------------------------------------------------------------------------
# fig4: A0E contribution decomposition A -> B -> BH -> C.
# ---------------------------------------------------------------------------
def fig4_a0e_decomposition(a0e: dict) -> None:
    metrics = a0e["metrics"]
    labels = ["A_momentum", "B_optimizer_only", "BH_curvature_feedback", "C_full_tangent"]
    titles = [
        "momentum\n" + r"$A$",
        "optimizer\n" + r"$B$",
        "raw curv.\n" + r"$BH$",
        "full tang.\n" + r"$C$",
    ]
    colors = [C_A, C_B, C_BH, C_C]
    values = [metrics[label]["overall"]["mean_relative_error"] for label in labels]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    x = np.arange(4)
    ax.bar(x, values, width=0.62, color=colors, edgecolor="0.2", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(min(values) * 0.45, max(values) * 2.6)
    for xi, v in zip(x, values):
        ax.text(xi, v * 1.22, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)
    # Slope connectors between adjacent model bars with white-backed labels.
    pairs = [
        (0, 1, "optimizer memory\n" + rf"$-{values[0]-values[1]:.3g}$", 0.42, 0.42),
        (1, 2, "raw curvature\n" + rf"$+{values[2]-values[1]:.3g}$", 1.5, 0.75),
        (2, 3, "clip-JVP\n" + rf"$-{values[2]-values[3]:.3g}$", 2.5, 0.115),
    ]
    for x0, x1, text, xt, yt in pairs:
        ax.plot(
            [x0, x1], [values[x0], values[x1]],
            color="0.45", linewidth=1.0, zorder=1,
        )
        ax.text(
            xt, yt, text, ha="center", va="center", fontsize=6.6, color="0.2",
            zorder=3, bbox=dict(fc="white", ec="none", pad=1.0),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=7)
    ax.set_ylabel(r"mean relative error (log scale)")
    ax.grid(True, which="major", axis="y")
    ax.text(
        -0.20, 1.04, "(a)", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="left",
    )
    _save(fig, "fig4_a0e_decomposition")


# ---------------------------------------------------------------------------
# fig5: A0F2 paired per-state B-to-C improvement for H=2,4,8.
# ---------------------------------------------------------------------------
def fig5_a0f2_improvement(a0f2: dict) -> None:
    rows = a0f2["results"]
    paired = a0f2["paired_b_c"]["per_horizon"]
    hs = (2, 4, 8)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    rng = np.random.default_rng(20260912)
    wins = []
    for i, h in enumerate(hs):
        group = [r for r in rows if int(r["H"]) == h]
        imp = np.asarray([
            float(r["relative_error_optimizer_only"])
            - float(r["relative_error_full_tangent"])
            for r in group
        ])
        jitter = rng.uniform(-0.13, 0.13, size=imp.size)
        ax.plot(
            np.full_like(imp, i + 1) + jitter, imp, linestyle="none",
            marker="o", markersize=3.2, color=H_COLORS[h], alpha=0.45,
            markerfacecolor="white", markeredgewidth=0.7,
        )
        mean = imp.mean()
        std = imp.std(ddof=1)
        ax.errorbar(
            [i + 1], [mean], yerr=[std], marker="s", color=H_COLORS[h],
            markerfacecolor=H_COLORS[h], markeredgewidth=0.8, capsize=3,
            elinewidth=1.0, markersize=4.5, zorder=5,
        )
        entry = paired[str(h)]["relative_error"]
        wins.append((i + 1, H_COLORS[h], entry["wins"], len(group)))
    ax.margins(y=0.07)
    for xi, color, win, total in wins:
        ax.text(
            xi, 0.955, rf"{win}/{total} wins",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.0, color=color,
        )
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlim(0.55, 3.45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([rf"$H={h}$" for h in hs])
    ax.set_ylabel(r"paired improvement  $\mathrm{err}_B-\mathrm{err}_C$")
    ax.set_xlabel(r"horizon $H$")
    ax.grid(True, which="major", axis="y")
    ax.text(
        -0.20, 1.04, "(a)", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="left",
    )
    _save(fig, "fig5_a0f2_improvement")


# ---------------------------------------------------------------------------
FIGURES_MD = """# FR Paper Figures v01

Rendered from frozen artifacts only (no observation regenerated):

| figure | data source (protocol) |
|---|---|
| fig1_a0d_epsilon_convergence | `response_propagation_v02_epsilon/summary.json` (FR_A0D_EPSILON_CONVERGENCE_V01) |
| fig2_a0f2_parity, fig3_a0f2_error_ladder, fig5_a0f2_improvement | `cross_partition_extension_v01/summary.json` (FR_A0F2_CROSS_PARTITION_EXTENSION_V01) |
| fig4_a0e_decomposition | `curvature_decomposition_v01/summary.json` (FR_A0E_CURVATURE_DECOMPOSITION_V01) |

Each figure ships as vector `.pdf` (fonttype 42, editable text) and 600-dpi `.png`.
Sizes: 7.0 in wide (full column) or 3.4 in wide (half column); STIX serif math.

## fig1_a0d_epsilon_convergence (a-f)
English caption: *Finite-difference convergence toward the tangent prediction.*
Stream-mean central finite differences $\\chi_{\\mathrm{FD}}(\\epsilon)$ of the
two-microbatch loss response (top row) and their relative error against the
frozen full-tangent value $\\chi_{\\mathrm{tangent}}$ (bottom row, dotted lines
are $O(\\epsilon^{2})$ guides), shown for the three locked v07 states
(columns; registered $\\epsilon_{\\mathrm{reg}}$ annotated) and horizons
$H=2,4,8$ (colors). Halving $\\epsilon$ contracts the FD change in 7/9
state-horizon cells; the pooled mean relative error falls from $2.19\\times10^{{-2}}$
at $\\epsilon$ to $4.52\\times10^{{-3}}$ at $\\epsilon/4$.
Note: the registered A0D scan is $\\{{\\epsilon, \\epsilon/2, \\epsilon/4\\}}$ at
each state's registered amplitude ($\\epsilon_{\\mathrm{reg}}\\approx 0.343,
0.0139, 0.0407$); no artifact scans $[10^{{-6}},10^{{-1}}]$.

## fig2_a0f2_parity (a,b)
English caption: *Prediction vs observation over the full v07 census.*
Per-state stream-mean predictions of the optimizer-only tangent $B$ (a) and
the full tangent $C$ (b) against the registered finite-difference response
$\\chi_{\\mathrm{FD}}$ for all 56 states and horizons $H=1,2,4,8$ (colors);
dashed line is $y=x$. $B$ degrades with horizon while $C$ stays on the diagonal
(pooled $R^{2}=0.9978$ and mean relative error $5.9\\times10^{{-3}}$ for $C$,
vs $R^{2}=0.7318$ and $0.128$ for $B$, over all 224 census state-horizon rows).

## fig3_a0f2_error_ladder
English caption: *Error ladder of the three response models.* Mean relative
error ($\\pm$ std across the 56 census states) of the momentum-only model $A$,
the optimizer-only tangent $B$, and the full tangent $C$ at horizons
$H=1,2,4,8$ (log scale). $A$ grows with $H$; $B$ accumulates optimizer-error
and reaches $0.30$ at $H=8$; $C$ stays at or below $6.1\\times10^{{-3}}$.

## fig4_a0e_decomposition
English caption: *Decomposition of the response-prediction error.* Mean
relative error (log scale; slope connectors mark paired contributions) over
the three FR_A0C states and $H=2,4,8$: optimizer memory removes
$6.36-0.165=6.20$ of error from $A$ to $B$; raw curvature feedback
$\\delta g=H\\delta\\theta$ without the clipping Jacobian *increases* the
error to $0.313$; coupling curvature to the clip Jacobian ($C$) removes
$0.286$, leaving $0.027$.

## fig5_a0f2_improvement
English caption: *Paired per-state improvement of the full tangent.*
For each census state, relative-error improvement
$\\mathrm{{err}}_B-\\mathrm{{err}}_C$ at $H=2,4,8$ (light points; squares are
mean $\\pm$ std over 56 states). All 168 non-control comparisons are wins
(zero losses); the mean improvement grows from $0.040$ at $H=2$ to $0.299$ at
$H=8$.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="*", default=None,
                        help="subset of figure names to render (default: all)")
    args = parser.parse_args()

    a0d = _load(A0D_SUMMARY)
    a0e = _load(A0E_SUMMARY)
    a0f2 = _load(A0F2_SUMMARY)

    registry = {
        "fig1_a0d_epsilon_convergence": lambda: fig1_a0d_epsilon_convergence(a0d),
        "fig2_a0f2_parity": lambda: fig2_a0f2_parity(a0f2),
        "fig3_a0f2_error_ladder": lambda: fig3_a0f2_error_ladder(a0f2),
        "fig4_a0e_decomposition": lambda: fig4_a0e_decomposition(a0e),
        "fig5_a0f2_improvement": lambda: fig5_a0f2_improvement(a0f2),
    }
    selected = args.figures if args.figures else list(registry)
    for name in selected:
        registry[name]()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "FIGURES.md").write_text(FIGURES_MD, encoding="utf-8")
    print(f"INDEX = {OUT_DIR / 'FIGURES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
