#!/usr/bin/env bash
# F3.5 sequential driver (tmux-hosted). Per cell: traj -> adjoint ->
# eig RAW -> eig DIML -> delete that cell's runtime cache; then postprocess.
# RAM-bounded: exactly ONE heavy eig process at a time (~65-70 GB peak).
# Resume-safe: every stage skips itself if its complete raw JSON exists.
set -u
cd /home/b205/projects/training_dynamics_lab/optimizer_sensitivity
PY=/home/b205/miniforge3/envs/minimind-exp/bin/python
OUT=outputs/f3_5_mode_coupling
GPU=${GPU:-0}
mkdir -p "$OUT/monitor"
echo $$ > "$OUT/driver.pid"
{ echo "=== f3.5 driver start $(date) gpu=$GPU ==="; date; uptime; free -h
  df -h /; nvidia-smi; } >> "$OUT/monitor/driver.log" 2>&1
( while sleep 300; do
    { date '+%F %T'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader 2>/dev/null
      free -m | awk 'NR==2{print "ram_used_MiB",$3}'; } \
      >> "$OUT/monitor/driver.mon" 2>/dev/null
  done ) &
MONPID=$!
trap 'kill $MONPID 2>/dev/null; rm -f "$OUT/driver.pid"' EXIT

stage() {  # name, extra args...
  local name=$1; shift
  echo "[f3.5 $(date '+%F %T')] START $name $*"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/f3_5_mode_coupling.py "$@" \
      >> "$OUT/${name}.out" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[f3.5 $(date '+%F %T')] FAIL $name rc=$rc"; return $rc
  fi
  echo "[f3.5 $(date '+%F %T')] DONE $name"
}

rc_all=0
for cell in 2042:26 2043:26 2042:64; do
  stage traj   traj   --cell "$cell" || { rc_all=1; break; }
  stage adjoint adjoint --cell "$cell" || { rc_all=1; break; }
  stage eigRAW eig    --cell "$cell" --metric RAW || { rc_all=1; break; }
  stage eigDIML eig   --cell "$cell" --metric DIML || { rc_all=1; break; }
  rm -rf "$OUT/runtime/${cell/:/_}"
  echo "[f3.5 $(date '+%F %T')] runtime cache released for $cell"
done
[ $rc_all -eq 0 ] && stage post postprocess
echo "[f3.5 $(date '+%F %T')] STREAM COMPLETE rc=$rc_all" | tee -a "$OUT/monitor/driver.log"
exit $rc_all
