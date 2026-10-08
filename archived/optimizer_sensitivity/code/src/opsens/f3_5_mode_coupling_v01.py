"""F3.5 unstable-mode coupling library (v01) — frozen scope in
docs/F3_5_UNSTABLE_MODE_COUPLING_PREREG.md (2026-09-29, commit 2ac9d45).

DIMENSIONLESS metric scaling (S^{-1} J S), F3-faithful Arnoldi with
eigenpair extraction (m=16, 2-pass reorthogonalized MGS, seed 91011),
conjugate-pair -> real invariant 2D bases, left/right eigenspaces,
principal angles, incremental propagated subspace overlap, biorthogonal
spectral projector with condition-number guard, alignment metrics.

Red lines: no optimizer/stabilizer/controller development; frozen F3
artifacts never modified; all thresholds frozen in the prereg before any
F3.5 eigenspace computation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from .f3_curvature_mechanism_v01 import (
    F3_ARNOLDI_M as F35_ARNOLDI_M, state_inner, state_norm, state_scale,
    state_add, zeros_like_state)
from .f3_curvature_mechanism_v01 import StepOperator  # noqa: F401 (re-export)

# ---- frozen constants (prereg §1-§11) --------------------------------------
F35_EPS_SCALE = 1e-30
F35_K = 8
F35_UNSTABLE_MARGIN = 1.05
F35_PAIR_RES_TOL = 0.05
F35_ADJOINT_TOL = 1e-5
F35_ADJOINT_SEEDS_U = (92001, 92002, 92003)
F35_ADJOINT_SEEDS_V = (93001, 93002, 93003)
F35_KAPPA_MAX = 1e3
F35_EXCITE_RATIO = 2.0
F35_PERSIST_RATIO = 1.25
F35_OFFSETS = (0, 1, 3, 7, 15)
F35_ARNOLDI_SEED = 91011          # inherited from F3 (frozen there)
F35_METRICS = ("RAW", "DIML")
F35_ORTH_DROP_TOL = 1e-8          # rank-deficiency drop in orthonormalization


# ---- DIMENSIONLESS scaling (prereg §2) --------------------------------------
def state_scales(named, opt) -> Dict[str, Dict[str, torch.Tensor]]:
    """s_x[n] = sqrt(mean(x0[n]^2)) + eps_scale from the cell's INITIAL frozen
    optimizer state (theta = current params at restore, m/v = optimizer state).
    Fixed for the whole H<=32 window; never re-estimated."""
    groups = {}
    for grp in opt.param_groups:
        for p in grp["params"]:
            groups[p] = grp
    out = {"th": {}, "m": {}, "v": {}}
    for name, p in named:
        st = opt.state[p]
        for ch, t0 in (("th", p.detach()), ("m", st["exp_avg"].detach()),
                       ("v", st["exp_avg_sq"].detach())):
            s = float(torch.sqrt(torch.mean(t0.double().square())).item())
            out[ch][name] = s + F35_EPS_SCALE
    return out


def scale_state(dz, scales, inverse: bool = False):
    """z~ = S^{-1} z  (inverse=False)  or  z = S z~  (inverse=True)."""
    fn = (lambda t, s: t / s) if not inverse else (lambda t, s: t * s)
    return {ch: {n: fn(t, scales[ch][n]) for n, t in dz[ch].items()}
            for ch in ("th", "m", "v")}


class ScaledOperator:
    """J~ = S^{-1} J S and its Euclidean adjoint S J^T S^{-1} (S real diagonal)."""

    def __init__(self, op, scales):
        self.op = op
        self.scales = scales

    def jvp(self, x):
        return scale_state(self.op.jvp(scale_state(x, self.scales, inverse=True)),
                           self.scales)

    def vjp(self, x):
        return scale_state(self.op.vjp(scale_state(x, self.scales)),
                           self.scales, inverse=True)

    @property
    def enabled(self):
        return self.op.enabled


# ---- random states -----------------------------------------------------------
def _random_unit_like(named, seed: int):
    """Seed-frozen random unit full-state vector (same numeric vector in every
    metric's coordinates; ERRATUM: scaling the RAW draw by S^{-1} would let the
    smallest-scale channel dominate by ~1e15 and degenerate the fp64 MGS)."""
    gen = torch.Generator().manual_seed(seed)
    dz = zeros_like_state(named, dtype=torch.float64)
    for ch in ("th", "m", "v"):
        for name, p in named:
            dz[ch][name] = torch.randn(p.shape, generator=gen, dtype=torch.float64)
    return state_scale(dz, 1.0 / state_norm(dz))


# ---- Arnoldi with eigenpair extraction (F3-faithful loop, prereg §1/§4) ------
def arnoldi_basis(matvec, named, m=F35_ARNOLDI_M, seed=F35_ARNOLDI_SEED):
    """Bit-faithful replica of f3.arnoldi_rho's loop (same op order), keeping
    the basis. Returns (H (m_eff+1, m_eff) float64 cpu, basis list, m_eff)."""
    basis = [_random_unit_like(named, seed)]
    H = torch.zeros((m + 1, m), dtype=torch.float64)
    m_eff = m
    for j in range(m):
        w = matvec(basis[j])
        # modified Gram-Schmidt with one reorthogonalization pass
        for _ in range(2):
            for i in range(j + 1):
                hij = state_inner(w, basis[i])
                H[i, j] += hij
                w = state_add(w, basis[i], sa=1.0, sb=-hij)
        h_next = state_norm(w)
        H[j + 1, j] = h_next
        if h_next <= 0.0:
            m_eff = j + 1
            break
        if j + 1 <= m - 1:
            basis.append(state_scale(w, 1.0 / h_next))
    return H[:m_eff + 1, :m_eff], basis[:m_eff], m_eff


def extract_pairs(H, basis, named, k=F35_K, tol=F35_PAIR_RES_TOL):
    """Top-k reliable eigenpairs by |lambda|. Conjugate pairs are represented
    ONCE (Im(lambda) > 0) by their real invariant 2D basis {Re v, Im v},
    jointly normalized; real eigenvalues by their (real) eigenvector.
    Per-pair residual: |h_{m+1,m} * e_m^T y_j| / |lambda_j| (F3 convention)."""
    import numpy as np

    m_eff = H.shape[1]
    Hm = H[:m_eff, :m_eff].numpy()
    vals, vecs = np.linalg.eig(Hm)
    h_last = float(H[m_eff, m_eff - 1]) if m_eff >= 1 else 0.0
    cands = []
    for j in range(m_eff):
        lam = complex(vals[j])
        resid = abs(h_last * vecs[m_eff - 1, j]) / max(abs(lam), 1e-300)
        cands.append((abs(lam), lam, j, resid))
    cands.sort(key=lambda t: -t[0])
    used = set()
    pairs = []
    for absl, lam, j, resid in cands:
        if len(pairs) >= k:
            break
        if j in used or resid > tol:
            continue
        y = vecs[:, j]
        if abs(lam.imag) <= 1e-12 * max(absl, 1e-300):
            used.add(j)
            re = _combine(basis, y.real)
            n = state_norm(re)
            if n <= 0.0:
                continue
            pairs.append({"kind": "real", "lam": complex(lam.real, 0.0),
                          "resid": float(resid), "re": state_scale(re, 1.0 / n),
                          "im": None})
        else:
            # find the conjugate partner among remaining candidates
            jc = None
            for absl2, lam2, j2, res2 in cands:
                if j2 in used or j2 == j:
                    continue
                if abs(lam2 - lam.conjugate()) <= 1e-8 * max(absl, 1e-300) \
                        and abs(lam2.imag) > 0:
                    jc = j2
                    break
            used.add(j)
            if jc is not None:
                used.add(jc)
            re = _combine(basis, y.real)
            im = _combine(basis, y.imag)
            n = math.sqrt(state_norm(re) ** 2 + state_norm(im) ** 2)
            if n <= 0.0:
                continue
            pairs.append({"kind": "cplx",
                          "lam": complex(lam.real, abs(lam.imag)),
                          "resid": float(resid),
                          "re": state_scale(re, 1.0 / n),
                          "im": state_scale(im, 1.0 / n)})
    return pairs


def _combine(basis, coeffs):
    """Ritz vector sum_k coeffs[k] * basis[k]. NOTE: builds FRESH tensors per
    (channel, name) -- f3.zeros_like_state shares tensor objects across the
    th/m/v dict entries, so in-place writes there would leak across channels."""
    out = {"th": {}, "m": {}, "v": {}}
    for ch in ("th", "m", "v"):
        first = basis[0][ch]
        for n, t0 in first.items():
            acc = coeffs[0] * t0 if coeffs[0] != 0.0 else torch.zeros_like(t0)
            for c, b in zip(coeffs[1:], basis[1:]):
                if c == 0.0:
                    continue
                acc = acc + c * b[ch][n]
            out[ch][n] = acc
    return out


def pair_dim(pair) -> int:
    return 2 if pair["kind"] == "cplx" else 1


def unstable_pairs(pairs, margin=F35_UNSTABLE_MARGIN):
    return [p for p in pairs if abs(p["lam"]) > margin]


def pair_basis(pairs) -> List[dict]:
    """Real basis vectors of the span: for cplx pairs {Re, Im}, real {v}."""
    out = []
    for p in pairs:
        out.append(p["re"])
        if p["kind"] == "cplx":
            out.append(p["im"])
    return out


def pair_module_shares(pair, named, modulewise_shares_fn):
    """Module energy shares on the theta channel, Re+Im combined norm."""
    t = {}
    for n, p in named:
        re = pair["re"]["th"][n]
        if pair["im"] is None:
            t[n] = re.abs()
        else:
            t[n] = torch.sqrt(re.double().square()
                              + pair["im"]["th"][n].double().square())
    return modulewise_shares_fn(named, t)


def extract_pairs_streaming(H, basis, named, modulewise_shares_fn,
                            k=F35_K, tol=F35_PAIR_RES_TOL,
                            margin=F35_UNSTABLE_MARGIN):
    """Memory-streaming variant of extract_pairs for the 100M-scale job:
    per-pair summary rows (eigenvalue / residual / module shares) are emitted
    and each pair's full-size tensors dropped immediately; the unstable span
    is orthonormalized INCREMENTALLY (MGS in |lambda| order) as pairs arrive.
    Peak RSS = Arnoldi basis + one pair's tensors + accumulated unstable
    basis. Returns (rows, U_list, unstable_lams). Same kept-set selection as
    extract_pairs (reliable top-k by |lambda|, one representative per
    conjugate pair)."""
    import numpy as np

    m_eff = H.shape[1]
    Hm = H[:m_eff, :m_eff].numpy()
    vals, vecs = np.linalg.eig(Hm)
    h_last = float(H[m_eff, m_eff - 1]) if m_eff >= 1 else 0.0
    cands = [(abs(complex(vals[j])), j) for j in range(m_eff)]
    cands.sort(key=lambda t: -t[0])
    used = set()
    rows, U, unst_lams = [], [], []
    kept = 0
    for absl, j in cands:
        if kept >= k:
            break
        lam = complex(vals[j])
        if j in used:
            continue
        resid = abs(h_last * vecs[m_eff - 1, j]) / max(absl, 1e-300)
        if resid > tol:
            continue
        y = vecs[:, j]
        is_cplx = abs(lam.imag) > 1e-12 * max(absl, 1e-300)
        if is_cplx:
            jc = None
            for absl2, j2 in cands:
                if j2 in used or j2 == j:
                    continue
                if abs(complex(vals[j2]) - lam.conjugate()) <= \
                        1e-8 * max(absl, 1e-300) and abs(vals[j2].imag) > 0:
                    jc = j2
                    break
            used.add(j)
            if jc is not None:
                used.add(jc)
            re = _combine(basis, y.real)
            im = _combine(basis, y.imag)
            kind = "cplx"
            lam = complex(lam.real, abs(lam.imag))
        else:
            used.add(j)
            re = _combine(basis, y.real)
            im = None
            kind = "real"
            lam = complex(lam.real, 0.0)
        n = state_norm(re) if im is None else math.sqrt(
            state_norm(re) ** 2 + state_norm(im) ** 2)
        if n <= 0.0:
            continue
        kept += 1
        # module shares (scale-invariant) on the theta channel, then drop
        t = {}
        for nn, p in named:
            s = re["th"][nn].double().square()
            if im is not None:
                s = s + im["th"][nn].double().square()
            t[nn] = torch.sqrt(s)
        shares = modulewise_shares_fn(named, t)
        top = max(shares, key=shares.get) if shares else ""
        rows.append({"idx": kept - 1, "kind": kind, "lam_re": lam.real,
                     "lam_im": lam.imag, "abs_lam": abs(lam),
                     "resid": float(resid),
                     "unstable": bool(absl > margin),
                     "top_module": top, "shares": shares})
        if absl > margin:
            unst_lams.append(lam)
            vecs_pair = [state_scale(re, 1.0 / n)] if im is None else \
                [state_scale(re, 1.0 / n), state_scale(im, 1.0 / n)]
            for v in vecs_pair:
                w = v
                for q in U:
                    c = state_inner(w, q)
                    w = state_add(w, q, sa=1.0, sb=-c)
                nw = state_norm(w)
                if nw > F35_ORTH_DROP_TOL * state_norm(v):
                    U.append(state_scale(w, 1.0 / nw))
        del re, im
    return rows, U, unst_lams


# ---- subspace utilities -------------------------------------------------------
def orthonormalize(states: List[dict], drop_tol=F35_ORTH_DROP_TOL) -> List[dict]:
    """Modified Gram-Schmidt over real state vectors; drops directions whose
    residual norm falls below drop_tol (rank deficiency guard)."""
    out = []
    for s in states:
        w = s
        for q in out:
            c = state_inner(w, q)
            w = state_add(w, q, sa=1.0, sb=-c)
        n = state_norm(w)
        n0 = state_norm(s)
        if n <= drop_tol * max(n0, 1e-300):
            continue
        out.append(state_scale(w, 1.0 / n))
    return out


def project_norm(Q: List[dict], z: dict) -> float:
    """|| Q^T z || for orthonormal Q."""
    return math.sqrt(sum(state_inner(q, z) ** 2 for q in Q))


def alignment(Q: List[dict], z: dict) -> Optional[float]:
    nz = state_norm(z)
    if nz <= 0.0 or not Q:
        return None
    return project_norm(Q, z) / nz


def gram_matrix(A: List[dict], B: List[dict]):
    import numpy as np
    G = np.zeros((len(A), len(B)), dtype=np.float64)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            G[i, j] = state_inner(a, b)
    return G


def principal_angles(A: List[dict], B: List[dict]):
    """(singular_values, theta_max, theta_mean) of the two subspaces."""
    import numpy as np
    if not A or not B:
        return None, None, None
    s = np.linalg.svd(gram_matrix(A, B), compute_uv=False)
    theta = np.arccos(np.clip(s, -1.0, 1.0))
    return s, float(theta.max()), float(theta.mean())


def mean_singulars(A: List[dict], B: List[dict]) -> Optional[float]:
    import numpy as np
    if not A or not B:
        return None
    s = np.linalg.svd(gram_matrix(A, B), compute_uv=False)
    return float(s.mean())


# ---- biorthogonal spectral projector (prereg §9) ------------------------------
def biorthogonal_projector(V: List[dict], W: List[dict]):
    """P = V (W^T V)^{-1} W^T with orthonormalized V, W (same dim d).
    Returns (kappa, apply) or (None, None) if dims mismatch/singular."""
    import numpy as np
    if len(V) == 0 or len(V) != len(W):
        return None, None
    G = gram_matrix(W, V)              # d x d
    sv = np.linalg.svd(G, compute_uv=False)
    if sv.min() <= 0.0:
        return float("inf"), None
    kappa = float(sv.max() / sv.min())
    Ginv_GT = np.linalg.solve(G, np.eye(len(V)))   # (W^T V)^{-1}

    def apply(z: dict) -> dict:
        coeffs = [sum(Ginv_GT[i, j] * state_inner(W[j], z)
                      for j in range(len(W))) for i in range(len(V))]
        out = {ch: {n: torch.zeros_like(t) for n, t in z[ch].items()}
               for ch in ("th", "m", "v")}
        for c, v in zip(coeffs, V):
            if c == 0.0:
                continue
            for ch in ("th", "m", "v"):
                for n, t in v[ch].items():
                    out[ch][n] += c * t
        return out

    return kappa, apply


def split_projection(P_apply, q: dict):
    """q_unstable = P q ; q_removed = q - P q (identity q = q_u + q_r exact)."""
    qu = P_apply(q)
    qr = state_add(q, qu, sa=1.0, sb=-1.0)
    return qu, qr


# ---- adjoint consistency gate (prereg §3) -------------------------------------
def adjoint_discrepancy(op, named, seed_u: int, seed_v: int):
    u = _random_unit_like(named, seed_u)
    v = _random_unit_like(named, seed_v)
    jv = op.jvp(v)
    ju = op.vjp(u)
    lhs = state_inner(u, jv)
    rhs = state_inner(ju, v)
    return abs(lhs - rhs) / max(abs(lhs), 1e-30), lhs, rhs


def adjoint_gate(op, named, seeds_u=F35_ADJOINT_SEEDS_U,
                 seeds_v=F35_ADJOINT_SEEDS_V, tol=F35_ADJOINT_TOL):
    rows = []
    for su, sv in zip(seeds_u, seeds_v):
        d, lhs, rhs = adjoint_discrepancy(op, named, su, sv)
        rows.append({"seed_u": su, "seed_v": sv, "lhs": lhs, "rhs": rhs,
                     "disc": d, "pass": bool(d <= tol)})
    return rows, all(r["pass"] for r in rows)


# ---- trajectory summary helpers (prereg §5/§6/§7) ------------------------------
def traj_summary(values: Dict[int, Optional[float]]):
    """median / max / AUC (trapezoid over available t, normalized)."""
    import numpy as np
    pts = sorted((t, v) for t, v in values.items() if v is not None)
    if not pts:
        return {"median": None, "max": None, "auc": None, "n": 0}
    vs = [v for _, v in pts]
    ts = [t for t, _ in pts]
    if len(pts) >= 2:
        auc = float(np.trapz(vs, ts) / (ts[-1] - ts[0]))
    else:
        auc = float(vs[0])
    return {"median": float(np.median(vs)), "max": float(max(vs)),
            "auc": auc, "n": len(pts)}
