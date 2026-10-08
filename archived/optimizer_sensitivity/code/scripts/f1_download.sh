#!/bin/bash
# F1 checkpoint downloader (preregistered in docs/F1_1B_REPLICATION_PREREG.md §2).
# Resumable, retry-looped, SHA256-gated. Writes ONLY to optimizer_sensitivity/external/peteish1.
# Usage: nohup bash scripts/f1_download.sh >> outputs/f1_1b_replication/download.log 2>&1 &
set -u
BASE_URL="https://olmo-checkpoints.org/ai2-llm/peteish1"
OUT_DIR="/home/b205/projects/training_dynamics_lab/optimizer_sensitivity/external/peteish1"
ORDER=(
  "380000 model.pt 5939703038 ca2fa02c8192651c7280e848e96c69f954451f784f26b97e8b11385c54470fa7"
  "380000 optim.pt 11879442406 3e48789a515e32d6d44618fecc77efd896b1055ce467568e06a79eab9df3bae6"
  "950000 model.pt 5939703038 6e7b1690215915dfb9cf697eefd2a22aea1404752ad1cdc531d4c18cae04e40b"
  "950000 optim.pt 11879442406 ffcb8374c53f635df2fdd22aab11aa1067f19d1285fc6074f2cffd141ccacdb2"
  "1530000 model.pt 5939703038 d55f32683d56d1e61d0b919d265a611bebec864875db404bcb0642b0dd419120"
  "1530000 optim.pt 11879442406 58f7f01d73a1e7772568a70234759ef82fa3d65367b75a813dc5b60e17158884"
)

verify() {
  local file="$1" expected_sha="$2"
  local actual
  actual=$(sha256sum "$file" 2>/dev/null | awk '{print $1}')
  [ "$actual" = "$expected_sha" ]
}

for spec in "${ORDER[@]}"; do
  set -- $spec
  step="$1"; name="$2"; expected_size="$3"; expected_sha="$4"
  out="${OUT_DIR}/step${step}_${name}"
  if [ -f "$out" ] && [ "$(stat -c %s "$out" 2>/dev/null || echo 0)" -ge "$expected_size" ]; then
    if verify "$out" "$expected_sha"; then
      echo "[$(date -u +%FT%TZ)] ALREADY-COMPLETE+VERIFIED $out"
      continue
    fi
  fi
  echo "[$(date -u +%FT%TZ)] fetching step${step}_${name} (${expected_size} bytes)"
  for attempt in $(seq 1 2000); do
    size=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$size" -ge "$expected_size" ]; then
      if verify "$out" "$expected_sha"; then
        echo "[$(date -u +%FT%TZ)] COMPLETE+VERIFIED $out"
        break
      else
        echo "[$(date -u +%FT%TZ)] SHA MISMATCH after full size; deleting and restarting $out"
        rm -f "$out"
      fi
    fi
    curl -fSL -C - --speed-time 60 --speed-limit 10000 --connect-timeout 20 \
      -o "$out" "${BASE_URL}/step${step}-unsharded/${name}" >/dev/null 2>&1
    sleep 3
  done
  size=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$size" -ge "$expected_size" ] && verify "$out" "$expected_sha"; then
    echo "[$(date -u +%FT%TZ)] OK $out"
  else
    echo "[$(date -u +%FT%TZ)] FAILED-AFTER-RETRIES $out (size=$(stat -c %s "$out" 2>/dev/null || echo 0))"
  fi
done
echo "[$(date -u +%FT%TZ)] download loop finished"
