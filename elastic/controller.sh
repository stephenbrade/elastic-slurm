#!/usr/bin/env bash
# The login1 controller, restarted if it exits (2026-11-13). Run it in tmux:
#   tmux new-session -d -s elastic "bash ~/live-music-diffusion-models-sa3/elastic/controller.sh"
# Stop it for good: touch /proj/spans/brade/lmdm/runs/.controller_stop, then kill the tmux session.
# (If it stays down >10 min the el:watchdog cpu job alerts and ticks the runs instead.)
# CSAIL (2026-11-15): ELASTIC_SITE=csail bash <repo>/elastic/controller.sh  (EL/ROOT from the site file)
EL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/elastic.py
if [ "${ELASTIC_SITE:-bluevela}" = bluevela ]; then
  EL=/u/brade/live-music-diffusion-models-sa3/elastic/elastic.py
  ROOT=/proj/spans/brade/lmdm/runs
  PY=/usr/bin/python3
else
  PY=${ELASTIC_PY:-/usr/bin/python3}
  ROOT=$($PY -I -c "import json,sys; print(json.load(open(sys.argv[1]))['root'])" "$(dirname "$EL")/sites/$ELASTIC_SITE.json")
  export ELASTIC_SITE
fi
mkdir -p "$ROOT"
while :; do
  $PY -I "$EL" controller --interval 120 >> "$ROOT/controller.log" 2>&1
  rc=$?
  [ -e "$ROOT/.controller_stop" ] && exit 0
  if [ "$rc" = 75 ]; then sleep 300; continue; fi       # another controller on this node holds the lock
  echo "[controller.sh] controller exited rc=$rc at $(date -Is); restarting in 60 s" >> "$ROOT/controller.log" 2>/dev/null
  $PY -I "$EL" alert "${ELASTIC_SITE:-bluevela} login controller exited rc=$rc; controller.sh restarts it in 60 s" || true
  sleep 60
done
