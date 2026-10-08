#!/usr/bin/env bash
# One-line F3 progress summary (safe to run anytime; read-only).
cd "$(dirname "$0")/.." || exit 1
echo "== systemd =="
systemctl --user status f3-remaining --no-pager 2>/dev/null | grep -E "Active|Main PID" || echo "f3-remaining.service not running"
echo "== driver log (last 8) =="
tail -8 outputs/f3_curvature_mechanism/driver2.log 2>/dev/null || echo "(none yet)"
echo "== latest step per stage =="
for f in outputs/f3_curvature_mechanism/rq*_*.out; do
  [ -f "$f" ] || continue
  printf "%-28s %s\n" "$(basename "$f")" "$(grep '^\\[' "$f" | tail -1)"
done
echo "== raw files =="
ls outputs/f3_curvature_mechanism/raw/ | grep -v gate | grep -v state || true
