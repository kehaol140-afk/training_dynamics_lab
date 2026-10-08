#!/usr/bin/env bash
# F4-A sequential map driver (tmux/setsid-hosted). Unit-level DONE resume.
# Usage: run_f4a_driver.sh <tag> <gpu> <scale:seed:step>...
set -u
cd /home/b205/projects/training_dynamics_lab/optimizer_sensitivity
PY=/home/b205/miniforge3/envs/minimind-exp/bin/python
OUT=outputs/f4a_phase_map
mkdir -p "$OUT/monitor"

TAG=$1; GPU=$2; shift 2
{ echo "=== f4a $TAG start $(date) gpu=$GPU ==="; date; uptime; free -h; df -h /; nvidia-smi; } >> "$OUT/monitor/$TAG.log" 2>&1
rc_all=0
for spec in "$@"; do
  scale=${spec%%:*}; rest=${spec#*:}; seed=${rest%%:*}; step=${rest##*:}
  done_f="$OUT/raw/$scale/seed_$seed/step_$step/DONE"
  if [ -f "$done_f" ]; then echo "[f4a $TAG $(date '+%T')] SKIP $spec"; continue; fi
  echo "[f4a $TAG $(date '+%T')] START $spec"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/f4a_phase_map.py unit \
      --scale "$scale" --seed "$seed" --step "$step" \
      >> "$OUT/monitor/${TAG}_units.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[f4a $TAG $(date '+%T')] FAIL $spec rc=$rc"; rc_all=1; break; fi
done
echo "[f4a $TAG $(date '+%T')] STREAM COMPLETE rc=$rc_all" | tee -a "$OUT/monitor/$TAG.log"
exit $rc_all
