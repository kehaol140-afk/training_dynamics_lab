#!/usr/bin/env bash
# Resumable F3 driver, designed to run INSIDE a tmux session (survives agent
# session teardown; the 2026-09-28 setsid/nohup attempt did not).
#
# Usage: run_f3_driver.sh <tag> <gpu> <stage-spec>...
#   stage-spec: rq1|rq2|rq3|rq3p|rq4:<seed>:<step>
#   rq3p = priority five arms first (FULL,NO_ATTN_Q,ONLY_ATTN_Q,NO_MLP,
#          ONLY_MLP + SO twins) -> enables the interim report early.
#
# Safety properties:
#   - done markers in outputs/f3_curvature_mechanism/done/ make every stage
#     idempotent; rerunning the driver never recomputes completed stages;
#   - rq1/rq2/rq3 resume from their own checkpoints after interruption;
#   - each stage is integrity-checked (f3_check_raw.py; rq3p uses a subset
#     check) before its done marker is written;
#   - resource snapshot at start + 5-minute light sampling (PHASE 10);
#   - on failure the stream STOPS (no automatic retry loops).
set -u
TAG=$1; GPU=$2; shift 2
cd /home/b205/projects/training_dynamics_lab/optimizer_sensitivity
PY=/home/b205/miniforge3/envs/minimind-exp/bin/python
OUT=outputs/f3_curvature_mechanism
mkdir -p "$OUT/done" "$OUT/monitor"
echo $$ > "$OUT/driver_$TAG.pid"
{
  echo "=== driver $TAG start $(date) ==="
  date; uptime; free -h; df -h /; nvidia-smi
} >> "$OUT/monitor/$TAG.log" 2>&1
( while sleep 300; do
    { date '+%F %T'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
      free -m | awk 'NR==2{print "ram_used_MiB",$3}'; } >> "$OUT/monitor/$TAG.mon" 2>/dev/null
  done ) &
MONPID=$!
trap 'kill $MONPID 2>/dev/null; rm -f "$OUT/driver_$TAG.pid"' EXIT

subset_check() {  # rq3p: five priority arms + SO twins present at h=32
  local seed=$1 step=$2
  "$PY" - "$seed" "$step" <<'EOF'
import json, sys
from pathlib import Path
seed, step = sys.argv[1], sys.argv[2]
raw = json.load(Path(f"outputs/f3_curvature_mechanism/raw/rq3_{seed}_{step}.json").open())
frozen = json.load(Path("outputs/f3_compatibility/frozen_reference/baseline_summary.json").open())
losses = [r["loss"] for r in raw["steps"]]
ref = frozen["losses"][f"{seed}:{step}"]
assert losses == list(ref), "loss drift vs frozen reference"
last = raw["steps"][-1]["arms"]
want = {"FULL", "NO_ATTN_Q", "ONLY_ATTN_Q", "NO_MLP", "ONLY_MLP"}
want |= {f"SO|{a}" for a in want}
missing = want - set(last)
assert not missing, f"missing at h=32: {sorted(missing)}"
assert len(raw["steps"]) == 32
print("SUBSET CHECK PASS")
EOF
}

run_stage() {
  local spec=$1
  local rq=${spec%%:*}
  local rest=${spec#*:}
  local seed=${rest%%:*}
  local step=${rest##*:}
  local cell="${seed}_${step}"
  local marker="$OUT/done/${rq}_${cell}.done"
  local extra=()
  if [ -f "$marker" ]; then echo "[driver $TAG] $(date '+%F %T') SKIP $spec (done)"; return 0; fi
  if [ "$rq" = rq3p ]; then rq=rq3; extra=(--arms "FULL,NO_ATTN_Q,ONLY_ATTN_Q,NO_MLP,ONLY_MLP"); fi
  local outlog="$OUT/${rq}_${cell}.out"
  echo "[driver $TAG] $(date '+%F %T') START $spec (gpu $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/f3_curvature_mechanism.py \
      --rq "$rq" --cell "$seed:$step" ${extra+"${extra[@]}"} >> "$outlog" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[driver $TAG] $(date '+%F %T') FAIL $spec rc=$rc (see $outlog)"; return $rc
  fi
  if [ ${#extra[@]} -gt 0 ]; then
    if ! subset_check "$seed" "$step" >> "$outlog" 2>&1; then
      echo "[driver $TAG] $(date '+%F %T') SUBSET CHECK FAIL $spec"; return 1
    fi
  else
    if ! "$PY" scripts/f3_check_raw.py "$rq" "$seed" "$step" >> "$outlog" 2>&1; then
      echo "[driver $TAG] $(date '+%F %T') CHECK FAIL $spec"; return 1
    fi
  fi
  { date '+%F %T'; sha256sum "$OUT/raw/${rq}_${cell}.json"; } > "$marker"
  echo "[driver $TAG] $(date '+%F %T') DONE $spec"
  return 0
}

rc_all=0
for spec in "$@"; do
  run_stage "$spec" || { rc_all=1; echo "[driver $TAG] stopping stream on failure"; break; }
done
echo "[driver $TAG] $(date '+%F %T') STREAM COMPLETE rc=$rc_all" | tee -a "$OUT/monitor/$TAG.log"
exit $rc_all
