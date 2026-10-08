#!/usr/bin/env bash
# Detached driver for the remaining F3 stages (survives session teardown;
# launched with setsid nohup). Execution order per prereg §10 with
# f3_check_raw.py integrity checks between stages.
#
# Stage plan at launch time (2026-09-28):
#   EARLY 2042:26 (GPU0), EARLY 2043:26 (GPU1): rq3, rq4 (rq1/rq2 done+checked)
#   MIDDLE 2042:64 (GPU1, after EARLY): rq1, rq2, rq3, rq4
set -u
cd /home/b205/projects/training_dynamics_lab/optimizer_sensitivity
PY=/home/b205/miniforge3/envs/minimind-exp/bin/python
LOG=outputs/f3_curvature_mechanism/driver2.log
: > "$LOG"

run_stages() {  # cell dev rq...
  local cell=$1 dev=$2; shift 2
  local tag=${cell/:/_}
  for rq in "$@"; do
    echo "[$(date +%T)] START $rq $cell" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$dev "$PY" scripts/f3_curvature_mechanism.py \
        --rq "$rq" --cell "$cell" >> "outputs/f3_curvature_mechanism/${rq}_${tag}.out" 2>&1
    rc=$?
    echo "[$(date +%T)] END $rq $cell rc=$rc" >> "$LOG"
    if [ $rc -ne 0 ]; then
      echo "DRIVER FAIL: $rq $cell" >> "$LOG"; return 1
    fi
    if ! "$PY" scripts/f3_check_raw.py "$rq" "${cell%:*}" "${cell#*:}" >> "$LOG" 2>&1; then
      echo "DRIVER CHECK FAIL: $rq $cell" >> "$LOG"; return 1
    fi
  done
  echo "DRIVER DONE: $cell" >> "$LOG"
}

run_stages 2042:26 0 rq3 rq4 &
run_stages 2043:26 1 rq3 rq4 &
wait
echo "EARLY CELLS COMPLETE" >> "$LOG"
run_stages 2042:64 1 rq1 rq2 rq3 rq4
echo "ALL CELLS COMPLETE" >> "$LOG"
