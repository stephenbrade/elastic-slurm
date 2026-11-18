#!/usr/bin/env bash
# Elastic POOL worker (kind=pool, 2026-11-15): claims idempotent shards of $ELASTIC_RUN one at a time
# and runs them (elastic.py pool-worker). Never submitted by hand: `elastic.py` supplies the slot's
# partition/account/QoS/gres, --requeue, --signal=B:USR1@120 and ELASTIC_RUN / ELASTIC_HOME / ELASTIC_SITE.
# A signal (preemption warning, scancel) stops the current shard; its lock is released and the shard
# is claimed again by another worker. Shard commands write atomically, so a killed shard leaves no output.
set -uo pipefail
echo "=== elastic pool worker $SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} $(hostname) $(date -Is)" \
     "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
${ELASTIC_PY:-/usr/bin/python3} -I "$ELASTIC_HOME/elastic.py" pool-worker "$ELASTIC_RUN" &
W=$!
trap 'kill -TERM $W 2>/dev/null' USR1 USR2 TERM
wait $W; rc=$?
# wait returns early when a trapped signal arrives: wait for the worker to finish its cleanup
while kill -0 $W 2>/dev/null; do wait $W; rc=$?; done
echo "=== pool worker exit rc=$rc $(date -Is)"
exit $rc
