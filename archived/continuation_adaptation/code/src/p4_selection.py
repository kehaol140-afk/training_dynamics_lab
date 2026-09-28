"""p4_selection — Stage 1b 的 Arm A 乘子选择（docs/P4_PROTOCOL.md §5 冻结规则）。

纯函数 + 台账 IO，不做任何训练：
- 判据：final V_monitor NLL 最低者胜；
- 并列（差 ≤ 1e-6）：优先 1.0x，再取较小乘子；
- 边界扩张：最优落在边界乘子（0.5/2.0）且其扩张值（0.25/4.0）未运行 →
  返回 needs_expansion，补跑后重新选择；扩张自动且仅一次；
  最终选中值落在最外边界（0.25/4.0）→ boundary_unresolved=True（接受，不再扩格）；
- B/C/D 使用选中乘子，不另行调参。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

TIE_EPS = 1e-6
INITIAL_GRID = [0.5, 1.0, 2.0]
EXPANSION_OF = {0.5: 0.25, 2.0: 4.0}
ALLOWED_MULTIPLIERS = {0.25, 0.5, 1.0, 2.0, 4.0}
OUTER_BOUNDARY = {0.25, 4.0}


def _fmt(m: float) -> str:
    return f"{m:g}"


def select_best(results: Dict[float, float]) -> Dict:
    """results: {multiplier: final_monitor_nll}（均为已完成 run 的实测值）。

    返回 {selected_multiplier, boundary_unresolved, needs_expansion}；
    needs_expansion 非 None 时须补跑该乘子后重新调用本函数。
    """
    if not results:
        raise ValueError("无可用 run 结果")
    for m in results:
        if float(m) not in ALLOWED_MULTIPLIERS:
            raise ValueError(f"非法乘子 {m}（网格/扩张表外）")

    best_val = min(results.values())
    tied = sorted(float(m) for m, v in results.items() if v <= best_val + TIE_EPS)
    selected = 1.0 if 1.0 in tied else tied[0]

    needs_expansion: Optional[float] = None
    if selected in EXPANSION_OF and EXPANSION_OF[selected] not in results:
        needs_expansion = EXPANSION_OF[selected]

    boundary_unresolved = needs_expansion is None and selected in OUTER_BOUNDARY

    return {
        "selected_multiplier": selected,
        "boundary_unresolved": bool(boundary_unresolved),
        "needs_expansion": needs_expansion,
    }


def run_id_for(arm: str, lr_multiplier: float) -> str:
    """run_id 命名：A 网格含乘子；B/C1/C2/D 用臂名。"""
    if arm == "A":
        return f"A_lr{_fmt(lr_multiplier)}"
    return arm


def write_ledger(path: Path, cell_id: str, runs: List[Dict], selection: Dict) -> Dict:
    ledger = {
        "cell_id": cell_id,
        "runs": runs,
        "selected_multiplier": selection["selected_multiplier"],
        "boundary_unresolved": selection["boundary_unresolved"],
        "selection_criterion": "final_monitor_nll",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger


def load_ledger(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
