#!/usr/bin/env bash
# Generic elastic training job. Never submitted by hand: `elastic.py` supplies every resource
# flag (nodes, gres, --signal=B:USR1@600, --requeue, mail) and the env below.
#   ELASTIC_RUN      run dir (run.json, ckpt/, logs/)
#   ELASTIC_HOME     this directory (sbatch runs a spool copy of the script, so $0 is useless)
#   ELASTIC_ROLE     primary | probe
#   ELASTIC_PRIMARY  (probe only) the job this one replaces
#
# A preemption requeue reruns this script from the top with the same job id; `prepare` then picks
# the newest checkpoint, so resume needs nothing special. Signals reach only this shell (B:), so
# they are forwarded to the trainer: USR1 = checkpoint and carry on, USR2 = checkpoint and exit.
set -uo pipefail
EL="${ELASTIC_PY:-/usr/bin/python3} -I $ELASTIC_HOME/elastic.py"
JOB=$SLURM_JOB_ID
echo "=== elastic job $JOB role=$ELASTIC_ROLE restart=${SLURM_RESTART_COUNT:-0} $(hostname)" \
     "nodes=$SLURM_NNODES gpus/node=${SLURM_GPUS_ON_NODE:-?} $(date -Is)"

CHILD=""
# Multi-node (2026-11-17, CSAIL slurm 25.05): the trainer is under srun, and srun does NOT forward
# SIGUSR1/USR2: it died of the signal (rc 138) and no rank saved (s0_4x2_vs-l40s job 3095801). So
# USR1/USR2 go to every task of the job's steps via `scancel --signal` (no --batch: not this shell).
fwd() {
  echo "[job] got $1 -> trainer pid ${CHILD:-none}"
  [ -n "$CHILD" ] || return 0
  if [ "${NNODES:-1}" -gt 1 ] && [ "$1" != TERM ]; then
    scancel --signal="$1" "$JOB" || kill -s "$1" "$CHILD"
  else
    kill -s "$1" "$CHILD"
  fi
}
trap 'fwd USR1' USR1
trap 'fwd USR2' USR2
trap 'fwd TERM' TERM

# --- upgrade probe: take over from the running primary without losing its allocation first
# (Only on the first start: a requeued probe is already the primary and has nothing to take over.)
if [ "$ELASTIC_ROLE" = probe ] && [ -n "$ELASTIC_PRIMARY" ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then
  P=$ELASTIC_PRIMARY
  ST=$(squeue -h -j "$P" -o %T 2>/dev/null)
  $EL event "$ELASTIC_RUN" handoff_begin job="$JOB" from="$P" from_state="${ST:-gone}"
  if [ "$ST" = RUNNING ]; then
    scancel --signal=USR2 --batch "$P"
    TIMEOUT=$(/usr/bin/python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('policy',{}).get('handoff_timeout_s',900))" "$ELASTIC_RUN/run.json")
    T0=$SECONDS
    while [ -n "$(squeue -h -j "$P" -o %T 2>/dev/null)" ]; do
      if (( SECONDS - T0 > TIMEOUT )); then
        echo "[job] primary $P did not exit in ${TIMEOUT}s; cancelling it"; scancel "$P"; sleep 30; break
      fi
      sleep 10
    done
  elif [ -n "$ST" ]; then
    scancel "$P"; sleep 5
  fi
  $EL event "$ELASTIC_RUN" handoff_done job="$JOB" from="$P" waited_s="$SECONDS"
fi

# --- the run's start lock (2026-10-27). The controller races one queued job per allowed shape, so
# two can start between its ticks; exactly one may train. A holder that is RUNNING wins (a probe
# waits for its handoff instead); COMPLETING means a preempted holder is still dying, so wait; a
# holder that is gone or back in the queue left a stale lock (a hard kill skips the EXIT trap).
LOCK="$ELASTIC_RUN/.train_lock"
# Breaking a stale lock (2026-11-17, CSAIL s0_race2): two jobs started in the same second, both saw
# the requeued (PENDING) holder, and both did `rm -rf; mkdir` -- the second removed the first's fresh
# lock and two trainers wrote one ckpt dir for a minute (3096964 and 3096965). Now every lock carries
# a unique token, and breaking it is ONE rename to .train_lock.dead.<token>: of several breakers
# exactly one rename succeeds (the target then exists, so `mv -T` fails for the rest), and a breaker
# that read an old token can never rename a newer lock away. Dead locks are swept after an hour.
# An empty job file is a lock being created right now: wait for it (30 s at most).
make_lock() {
  mkdir "$LOCK" 2>/dev/null || return 1
  echo "$JOB" > "$LOCK/job"; echo "$JOB.$(date +%s%N).$$.$RANDOM" > "$LOCK/token"
}
break_stale() {   # $1 = the token of the lock we judged stale
  mv -T "$LOCK" "$LOCK.dead.$1" 2>/dev/null || true
  find "$(dirname "$LOCK")" -maxdepth 1 -name "$(basename "$LOCK").dead.*" -mmin +60 -exec rm -rf {} + 2>/dev/null
}
acquire_lock() {
  local i H HS TOK E=0
  for i in $(seq 1 60); do
    make_lock && return 0
    H=$(cat "$LOCK/job" 2>/dev/null); TOK=$(cat "$LOCK/token" 2>/dev/null)
    if [ -z "$H" ]; then       # being written, or a holder killed between mkdir and echo
      E=$((E + 1)); (( E >= 6 )) && { echo "[job] start lock with no holder for 30 s; taking it"; break_stale "${TOK:-nojob.$(stat -c %Y "$LOCK" 2>/dev/null)}"; }
      sleep 5; continue
    fi
    [ "$H" = "$JOB" ] && return 0                  # our own lock, from before a requeue
    HS=$(squeue -h -j "$H" -o %T 2>/dev/null)
    case "$HS" in
      RUNNING) [ "$ELASTIC_ROLE" = probe ] || return 1; sleep 10 ;;
      COMPLETING|SIGNALING|STAGE_OUT) sleep 10 ;;
      *) echo "[job] stale start lock from job ${H:-?} (${HS:-gone}); taking it"
         break_stale "${TOK:-legacy.$H.$(stat -c %Y "$LOCK/job" 2>/dev/null)}" ;;
    esac
  done
  # 10 min of COMPLETING is a holder stuck dying, not training: take the lock rather than give
  # this allocation back.
  if [ "$HS" = COMPLETING ]; then
    echo "[job] holder $H stuck COMPLETING; taking the start lock"
    break_stale "${TOK:-legacy.$H.$(stat -c %Y "$LOCK/job" 2>/dev/null)}"
    make_lock && return 0
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

# --- resolve accum + checkpoint for THIS allocation
PREP=$($EL prepare "$ELASTIC_RUN") || { echo "FATAL: prepare failed"; exit 1; }
eval "$PREP"
echo "=== nodes=$NNODES gpus/node=$GPN micro=$ELASTIC_MICRO_BS accum=$ACCUM global=$ELASTIC_GLOBAL_BS resume_step=$RESUME_STEP"

# The controller is a tmux process on a login node and can die silently: shout if it looks dead.
HB=$(dirname "$ELASTIC_RUN")/.controller_heartbeat
if [ ! -f "$HB" ] || (( $(date +%s) - $(stat -c %Y "$HB") > 1800 )); then
  $EL notify "$ELASTIC_RUN" "controller heartbeat stale (job $JOB started without it)"
fi

cd "$REPO" || exit 1

# --- multi-node preflight (2026-10-28): the venv's python is a symlink into /u/brade, and a node
# that cannot see it fails every rank there with `execve(): ... No such file or directory`
# (p5-r22-n4, abl_csail job 466313). Every node must import torch before anything else runs; a
# node that cannot is excluded for every run and this job re-races (exit 0 + `bad_node`, which the
# controller treats like a lost start race, not a crash or a finished run).
# Same srun also reports whether each node has an RDMA device (2026-11-17, CSAIL): NCCL picks
# NET/IB on a node with a RoCE NIC and NET/Socket on one without, and a job spanning both dies at
# init ("socketFinalizeAccept: wrong type 4 != 3", vs-v100 agrawal + freeman). If the nodes are
# MIXED, NCCL_IB_DISABLE=1 puts every rank on sockets; all-IB (Blue Vela) or all-socket is untouched.
# GPU_PREFLIGHT=1 (site gpu_count_preflight, CSAIL 2026-11-17) also runs it for single-node shapes and
# requires torch.cuda.device_count() == the GPUs Slurm gave this node: gpu20-2.drl showed torch 1 of 2,
# 3 of 4, 7 of 8 GPUs (one dead GPU), and the 2080 Ti nodes 0 (CUDA init fails), which the trainer only
# caught as a global-batch FATAL after loading. Such a node is a bad node like the one above. Node
# names are Slurm's ($SLURMD_NODENAME: CSAIL's "gpu20-2.drl" is not `hostname -s`).
if (( NNODES > 1 )) || [ "${GPU_PREFLIGHT:-0}" = 1 ]; then
  PF=$(srun --nodes="$NNODES" --ntasks-per-node=1 --cpu-bind=none bash -c \
          'h=${SLURMD_NODENAME:-$(hostname -s)}
           if [ "$3" = 1 ]; then n=$("$1" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
             [ "$n" = "$2" ] && echo "ok $h" || echo "bad $h torch_sees=${n:-none}_of_$2"
           else "$1" -c "import torch" >/dev/null 2>&1 && echo "ok $h" || echo "bad $h"; fi
           [ -n "$(ls /sys/class/infiniband 2>/dev/null)" ] && echo "ib $h" || echo "noib $h"' _ "$PYBIN" "$GPN" "${GPU_PREFLIGHT:-0}" 2>/dev/null)
  echo "$PF" | grep '^bad' || true
  BAD=$(echo "$PF" | awk '$1=="bad"{print $2}' | sort -u | paste -sd, -)
  NIB=$(echo "$PF" | awk '$1=="ib"' | sort -u | wc -l); NNOIB=$(echo "$PF" | awk '$1=="noib"' | sort -u | wc -l)
  if (( NNODES > 1 && NIB > 0 && NNOIB > 0 )) && [ -z "${NCCL_IB_DISABLE:-}" ]; then
    echo "[job] preflight: RDMA on $NIB node(s), none on $NNOIB: NCCL_IB_DISABLE=1 (sockets everywhere)"
    export NCCL_IB_DISABLE=1
    $EL event "$ELASTIC_RUN" nccl_ib_disabled job="$JOB" ib="$NIB" noib="$NNOIB"
  fi
  if [ -n "$BAD" ]; then
    echo "[job] preflight: node(s) $BAD cannot run $PYBIN on all their GPUs; excluding them and re-racing"
    $EL event "$ELASTIC_RUN" bad_node job="$JOB" nodes="$BAD"
    $EL exclude-nodes "$BAD"
    exit 0
  fi
fi

for c in "${PRE_LAUNCH[@]}"; do
  echo "=== pre-launch: $c"
  PYTHON="$PYBIN" bash -c "$c" || { echo "FATAL: pre-launch gate failed: $c"; exit 1; }
done

if (( NNODES > 1 )); then
  # One task per GPU; Lightning's SLURMEnvironment (auto_requeue off) takes the topology from srun.
  srun --kill-on-bad-exit=1 --cpu-bind=none --ntasks-per-node="$GPN" \
    "$PYBIN" -u "$SCRIPT" "${ARGS[@]}" &
else
  "$PYBIN" -u "$SCRIPT" "${ARGS[@]}" &
fi
CHILD=$!

# `wait` returns early (>128) whenever a trapped signal arrives; keep waiting until the trainer
# is really gone, then collect its status.
while :; do
  wait "$CHILD"; RC=$?
  kill -0 "$CHILD" 2>/dev/null && continue
  if (( RC > 128 )); then wait "$CHILD" 2>/dev/null; R2=$?; (( R2 != 127 )) && RC=$R2; fi
  break
done
$EL event "$ELASTIC_RUN" trainer_exit job="$JOB" rc="$RC"
exit "$RC"
