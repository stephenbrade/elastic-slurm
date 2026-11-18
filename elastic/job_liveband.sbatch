#!/usr/bin/env bash
# Elastic job for a kind=liveband run (2026-11-13): ~/liveband_repro sbatch/train_data_ddp.sh under
# the controller. Never submitted by hand: `elastic.py` supplies the resource flags (1 node, the
# run's GPUs, --requeue, --signal=B:USR1@600) and ELASTIC_RUN / ELASTIC_HOME / ELASTIC_ROLE.
#
# The trainer resumes from <out>/ckpt_last.pt by itself; `prepare` checks that file is finite first
# and refuses a from-scratch start of a run that has progress. USR1 / USR2 / TERM all mean
# "checkpoint and exit 0" to the LiveBand trainer and are forwarded to train_data_ddp.sh as USR1/TERM.
set -uo pipefail
EL="${ELASTIC_PY:-/usr/bin/python3} -I $ELASTIC_HOME/elastic.py"
JOB=$SLURM_JOB_ID
echo "=== elastic liveband job $JOB restart=${SLURM_RESTART_COUNT:-0} $(hostname)" \
     "gpus=${SLURM_GPUS_ON_NODE:-?} $(date -Is)"

CHILD=""
fwd() { echo "[job] got $1 -> train_data_ddp.sh pid ${CHILD:-none} as $2"; [ -n "$CHILD" ] && kill -s "$2" "$CHILD"; }
trap 'fwd USR1 USR1' USR1
trap 'fwd USR2 USR1' USR2
trap 'fwd TERM TERM' TERM

# --- start lock: exactly one trainer per run. Same protocol as job.sbatch; an adopted plain-sbatch
# job holds it by job id (`elastic.py adopt`), so this job loses the race while that one RUNS.
LOCK="$ELASTIC_RUN/.train_lock"
acquire_lock() {
  local i H HS
  for i in $(seq 1 60); do
    if mkdir "$LOCK" 2>/dev/null; then echo "$JOB" > "$LOCK/job"; return 0; fi
    H=$(cat "$LOCK/job" 2>/dev/null)
    [ "$H" = "$JOB" ] && return 0
    HS=$(squeue -h -j "$H" -o %T 2>/dev/null)
    case "$HS" in
      RUNNING) return 1 ;;
      COMPLETING|SIGNALING|STAGE_OUT) sleep 10 ;;
      *) echo "[job] stale start lock from job ${H:-?} (${HS:-gone}); taking it"; rm -rf "$LOCK" ;;
    esac
  done
  if [ "$HS" = COMPLETING ]; then
    echo "[job] holder $H stuck COMPLETING; taking the start lock"; rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null && { echo "$JOB" > "$LOCK/job"; return 0; }
  fi
  return 1
}
if ! acquire_lock; then
  echo "[job] lost the start race to job $(cat "$LOCK/job" 2>/dev/null); exiting"
  $EL event "$ELASTIC_RUN" lost_race job="$JOB" holder="$(cat "$LOCK/job" 2>/dev/null)"
  exit 0
fi
release_lock() { [ "$(cat "$LOCK/job" 2>/dev/null)" = "$JOB" ] && rm -rf "$LOCK"; }
trap release_lock EXIT

# train_data_ddp.sh test-only knobs must never leak in from the controller's environment
# (--export=ALL); run.json "env" may still set them on purpose.
unset TRAIN_PY SKIP_GATES WANDB LB_DUMP_BATCH OUT NGPU
PREP=$($EL prepare "$ELASTIC_RUN") || { echo "FATAL: prepare failed"; exit 1; }
eval "$PREP"
echo "=== out=$OUT resume_step(logged)=$RESUME_STEP args: ${ARGS[*]}"

HB=$(dirname "$ELASTIC_RUN")/.controller_heartbeat
if [ ! -f "$HB" ] || (( $(date +%s) - $(stat -c %Y "$HB") > 1800 )); then
  $EL notify "$ELASTIC_RUN" "controller heartbeat stale (job $JOB started without it)"
fi

cd "$REPO" || exit 1
for c in "${PRE_LAUNCH[@]}"; do
  echo "=== pre-launch: $c"
  bash -c "$c" || { echo "FATAL: pre-launch gate failed: $c"; exit 1; }
done

# train_data_ddp.sh sees SLURM_JOB_ID and runs inline (no self-submit); it runs its own gates.
bash "$SCRIPT" "${ARGS[@]}" &
CHILD=$!
while :; do
  wait "$CHILD"; RC=$?
  kill -0 "$CHILD" 2>/dev/null && continue
  if (( RC > 128 )); then wait "$CHILD" 2>/dev/null; R2=$?; (( R2 != 127 )) && RC=$R2; fi
  break
done
$EL event "$ELASTIC_RUN" trainer_exit job="$JOB" rc="$RC"
exit "$RC"
