# FR Paper Figures v01

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
Stream-mean central finite differences $\chi_{\mathrm{FD}}(\epsilon)$ of the
two-microbatch loss response (top row) and their relative error against the
frozen full-tangent value $\chi_{\mathrm{tangent}}$ (bottom row, dotted lines
are $O(\epsilon^{2})$ guides), shown for the three locked v07 states
(columns; registered $\epsilon_{\mathrm{reg}}$ annotated) and horizons
$H=2,4,8$ (colors). Halving $\epsilon$ contracts the FD change in 7/9
state-horizon cells; the pooled mean relative error falls from $2.19\times10^{{-2}}$
at $\epsilon$ to $4.52\times10^{{-3}}$ at $\epsilon/4$.
Note: the registered A0D scan is $\{{\epsilon, \epsilon/2, \epsilon/4\}}$ at
each state's registered amplitude ($\epsilon_{\mathrm{reg}}\approx 0.343,
0.0139, 0.0407$); no artifact scans $[10^{{-6}},10^{{-1}}]$.

## fig2_a0f2_parity (a,b)
English caption: *Prediction vs observation over the full v07 census.*
Per-state stream-mean predictions of the optimizer-only tangent $B$ (a) and
the full tangent $C$ (b) against the registered finite-difference response
$\chi_{\mathrm{FD}}$ for all 56 states and horizons $H=1,2,4,8$ (colors);
dashed line is $y=x$. $B$ degrades with horizon while $C$ stays on the diagonal
(pooled $R^{2}=0.9978$ and mean relative error $5.9\times10^{{-3}}$ for $C$,
vs $R^{2}=0.7318$ and $0.128$ for $B$, over all 224 census state-horizon rows).

## fig3_a0f2_error_ladder
English caption: *Error ladder of the three response models.* Mean relative
error ($\pm$ std across the 56 census states) of the momentum-only model $A$,
the optimizer-only tangent $B$, and the full tangent $C$ at horizons
$H=1,2,4,8$ (log scale). $A$ grows with $H$; $B$ accumulates optimizer-error
and reaches $0.30$ at $H=8$; $C$ stays at or below $6.1\times10^{{-3}}$.

## fig4_a0e_decomposition
English caption: *Decomposition of the response-prediction error.* Mean
relative error (log scale; slope connectors mark paired contributions) over
the three FR_A0C states and $H=2,4,8$: optimizer memory removes
$6.36-0.165=6.20$ of error from $A$ to $B$; raw curvature feedback
$\delta g=H\delta\theta$ without the clipping Jacobian *increases* the
error to $0.313$; coupling curvature to the clip Jacobian ($C$) removes
$0.286$, leaving $0.027$.

## fig5_a0f2_improvement
English caption: *Paired per-state improvement of the full tangent.*
For each census state, relative-error improvement
$\mathrm{{err}}_B-\mathrm{{err}}_C$ at $H=2,4,8$ (light points; squares are
mean $\pm$ std over 56 states). All 168 non-control comparisons are wins
(zero losses); the mean improvement grows from $0.040$ at $H=2$ to $0.299$ at
$H=8$.
