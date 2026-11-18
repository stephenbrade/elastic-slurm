# elastic/ — preemptible, resizable training on Blue Vela

Blue Vela gives us only `slurm_grp_preemptable` / QoS `burst` (partition `gpu-low`). Preemption is
`PreemptMode=REQUEUE` with a 10 min grace, and `send_user_signal` delivers our `--signal` at the start
of that grace. This directory turns that into: **save on the warning, resume on requeue, shrink when
the GPUs are not there, grow back when they are, mail you (site/run "mail" or $ELASTIC_MAIL) at every step.** The effective
batch is held fixed: `accum = global_bs / (micro_bs × nodes × gpus)`.

| file | runs where | role |
|---|---|---|
| `elastic.py` | login node (stdlib, py3.9) and inside jobs | CLI + controller + `prepare` |
| `job.sbatch` | compute | resolves accum + checkpoint, forwards signals, handoff, srun for multi-node |
| `smoke_train.py` | compute | toy trainer that fails if any optimizer step is not exactly `global_bs` samples |
| `../stable_audio_3/training/elastic.py` | inside the trainer | `ElasticCallback` (USR1 = save, USR2 = save+exit), `slurm_topology()` |

`scripts/train_lmdm.py` and `scripts/train_arc.py` both use the callback and topology.

## Start a run

```bash
R=/proj/spans/brade/lmdm/runs/<name>; mkdir -p $R
cat > $R/run.json <<'EOF'
{
  "name": "<name>",
  "repo": "/u/brade/live-music-diffusion-models-sa3",
  "python": "/proj/spans/brade/lmdm/envs/train/bin/python",
  "script": "scripts/train_arc.py",
  "args": ["...everything except --batch_size --accum_batches --save_dir --ckpt_path...",
           "--checkpoint_every", "500"],
  "global_bs": 160, "micro_bs": 20,
  "configs": [[2, 8], [1, 8], [1, 4], [1, 2]],
  "pre_launch": ["$PYTHON gates/<the gate for this stage>.py ..."],
  "env": {"WANDB_RUN_ID": "<fixed id>", "WANDB_RESUME": "allow", "HF_HOME": "/proj/spans/brade/lmdm/.cache/hf"},
  "slurm": {"cpus_per_gpu": 8, "mem_per_gpu_gb": 96, "time": "7-00:00:00"}
}
EOF
E="/usr/bin/python3 -I ~/live-music-diffusion-models-sa3/elastic/elastic.py"
$E start <name>            # submits the best fit now; --config I to force one
$E status                  # all runs + controller heartbeat
```

`configs` is a SET of allowed shapes, `[nodes, gpus_per_node]` or `[nodes, gpus_per_node, type]`;
list order no longer matters (2026-10-27). The controller ranks the shapes that can start on what is
free now: most GPUs, then the GPU type with the most spare whole nodes (up to `policy.headroom_cap`,
default 3), then `policy.gpu_preference` (default `["h100", "h200"]`). A PENDING job (a requeue
after preemption keeps its old request) switches to a better-ranked shape after `fit_patience_s`
even if its own shape still fits; a RUNNING job only ever moves for more GPUs. `elastic.py fit`
shows the ranking read-only. Every shape must divide `global_bs` (checked at `start`). Checkpoint policy agreed 2026-10-23: **ARC `--checkpoint_every 500`, all kept**; finetunes
`--checkpoint_every 1000 --checkpoint-keep 1 --checkpoint-milestone-every 10000`. A preemption costs
at most one optimizer step anyway, because of the USR1 save.

**For an ARC actor, the `--context-embed` rule in CLAUDE.md still applies.** Put the E2 check in
`pre_launch` so a resume can never quietly launch on a non-E2 init.

## The controller

Cron and scrontab are disabled for users, so it runs in tmux on **login1**, inside a restart loop
(`controller.sh`, 2026-11-13: it restarts the controller 60 s after any exit and sends an ntfy alert):

```bash
tmux new-session -d -s elastic "bash ~/live-music-diffusion-models-sa3/elastic/controller.sh"
```

To stop it for good: `touch /proj/spans/brade/lmdm/runs/.controller_stop`, then kill the session.
Otherwise the watchdog below takes over after 10 min.

### Watchdog (2026-11-13)

On 2026-11-13 at 06:48 UTC login1 lost /proj (jobs died 135:0 / 0:53). The controller died silently
with it, and **every run sat dead for ~8 h**; the 2026-10-26 and 2026-11-04 incidents looked the same.
`watchdog.sbatch` runs `elastic.py watchdog` as job `el:watchdog` on the **cpu partition**, away from
login1. Every 2 min it checks `.controller_heartbeat`. If the heartbeat is more than 10 min old, it
sends an ntfy alert (hourly) and runs one controller pass itself (`controller --once --role watchdog`,
heartbeat `.controller_heartbeat_watchdog`). It keeps one `--dependency=singleton` successor queued,
and the controller resubmits it every tick if none is queued, so each watches the other.

Exclusion is a heartbeat handover, NOT the flock: **GPFS does not enforce flock across nodes**
(tested: a cpu node acquired `.controller.lock` while login1 held it). The watchdog re-checks the
main heartbeat before each run and stops when it is fresh. A main controller that finds a fresh
watchdog heartbeat waits one interval + 60 s before its first pass. Compute nodes reach ntfy and
can submit jobs, but cannot ssh to login1, so restarting tmux after an outage still needs a person
(the alert gives the command). Stop the watchdog: `touch /proj/spans/brade/lmdm/runs/.watchdog_off`.

A tick that raises (e.g. an `sbatch` that fails during a resubmit) now sends an hourly ntfy alert per
run, `CONTROLLER ERROR ...`. Urgent alerts (halts, controller errors, every LiveBand resubmit) bypass
the 20/h cap.

Each tick it writes `runs/.controller_heartbeat`; a job that starts while that is >30 min stale
mails you. Policy knobs (`run.json` → `policy`, defaults in `elastic.py`): pending patience 10 min
before going smaller, 30 min dwell before trying bigger, 15 min probe patience, probe backoff 30
min → 2 h. `sbatch --test-only` is only a hint: here it says ~now+600 s for every size.

## LiveBand runs (`"kind": "liveband"`, 2026-11-13)

`~/liveband_repro` runs (`sbatch/train_data_ddp.sh`, 1 node x N GPUs DDP) are under the controller.
On 2026-11-13 `lb_slakh_ccdc_paper_d{0,1,11}` were plain `--requeue` jobs, and a requeue that
failed instantly (FAILED 0:53) left them dead with nothing to resubmit them. A LiveBand run dir in
`runs/` holds:

```json
{"name": "lb_slakh_ccdc_t3_adagp_d1", "kind": "liveband",
 "repo": "/u/brade/liveband_repro", "python": "/proj/spans/brade/lmdm/envs/train/bin/python",
 "script": "sbatch/train_data_ddp.sh",
 "args": ["--steps", "100000", "--d-channels", "1024", "--delta", "1", "--latent-dim", "128"],
 "out_dir": "/proj/spans/brade/liveband/runs/lb_slakh_ccdc_t3_adagp_d1",
 "env": {"RUN": "lb_slakh_ccdc_t3_adagp_d1", "DATA": "<latent root>", "WANDB_RUN_ID": "<RUN>"},
 "plain_job_names": ["lb_ccdc_t3_adagp"], "configs": [[1, 4, "h100"]],
 "slurm": {"cpus_per_gpu": 12, "mem_per_gpu_gb": 96, "time": "7-00:00:00"}}
```

`args` excludes `--out/--name/--data/--direction`, which train_data_ddp.sh sets from `env` (`OUT` = `out_dir`).
Jobs run `job_liveband.sbatch`, which takes the same `.train_lock`, runs `prepare` and then
train_data_ddp.sh inline (it runs its own gates). `prepare` puts `<out>/ckpt_last.pt` through
`scripts/check_ckpt_finite.py`. A non-finite file is **not** renamed, because the trainer would then
start from step 0; the run halts NONFINITE instead. `prepare` also refuses to start a run from
scratch if it already has progress. USR1/USR2/TERM all mean "save and exit 0" to the LiveBand trainer.

| case | controller |
|---|---|
| FAILED / NODE_FAIL / vanished (UNKNOWN) | resubmit, counted toward `max_failures` (2), then HALTED + ntfy |
| PREEMPTED / TIMEOUT, or COMPLETED below `--steps` (time limit, unrequeued preempt) | resubmit + ntfy |
| COMPLETED below `--steps` twice with no progress | counted as a failure |
| no new `step` record in `<out>/metrics.jsonl` for `stall_s` (600 s); `startup_stall_s` (1800 s) applies until the first record after the controller sees the job RUNNING, so a job that hangs right after it starts is cancelled after ~30 min, not 10 | scancel + resubmit |
| new `<out>/nonfinite_step*.pt` (trainer NaN stop) | HALTED NON-FINITE, nothing resubmitted |
| logged step >= `--steps` and COMPLETED | done |
| a job that starts and ends between two ticks | judged as above (not silently re-raced) |

**Adopting a live plain-sbatch job** (no double submit): `elastic.py adopt <run> <jobid> [--dry-run]`.
It checks the job's command, that its name is in `plain_job_names`, that a running job's log header
names `RUN`, and that no other live job trains that run. The job then becomes the primary and the
holder of the start lock. If Slurm requeues it, the controller waits. When it ends, the controller
resubmits under elastic. `start` refuses while any plain job for the run is live.

**Pause / resume:** `elastic.py pause <run>` (add `--save-stop` to checkpoint and stop a running job:
USR1, exit 0), or simply `scancel` the job, which the controller reads as a pause.
`elastic.py resume <run>` resubmits on the next tick, and the trainer resumes from `ckpt_last.pt`
after the finite check.

## Notifications

**Blue Vela cannot send email.** Port 25 is blocked, postfix is down, and Slurm's `MailProg` is
missing, so `--mail-type` mail is never delivered. The controller alerts on preemption, rescale,
crash, halt and finish through, in order:

1. **ntfy.sh push** (works; confirmed 2026-10-23). The topic is in `/proj/spans/brade/lmdm/.ntfy_topic`
   (0600). Subscribe to it in the ntfy phone app.
2. **SMTP on 587**, if `/proj/spans/brade/lmdm/.smtp.json` (0600) exists:
   `{"host","port","user","password","from"}`, e.g. a bot Gmail with an app password.
3. Slurm mail via a 1-second cpu job named with the message. Kept in case the admins fix the MTA.

`elastic.py test-notify --msg "..."` exercises all three and reports each one's result.

## NaN / corruption guards (2026-11-12)

`ss_bc1_lag4_jam_mix2stem_cenc` resumed from a NaN checkpoint at 22k and trained, checkpointed and fed
the S-DMD chain NaN to 100k. Nothing stopped it. Now:

| guard | where | what happens |
|---|---|---|
| loss stop | `ElasticCallback` (every trainer using it) | `NONFINITE_MAX_STEPS` (default 5, 0 = off) optimizer steps in a row with a non-finite loss: `nonfinite` event + `<run>/NONFINITE` marker, `NonFiniteError` on every rank, non-zero exit |
| save guard | `ElasticCallback.on_save_checkpoint`, `GenSnapshot` | any save (routine, milestone, elastic, gen snapshot) with a NaN/inf trainable param or a NaN `state_dict` tensor raises instead of writing |
| atomic writes | `AtomicCheckpointIO` via `slurm_topology()` | `<name>.tmp` then `os.replace`; `.tmp` is never resumed from, and the controller deletes ones older than 6 h |
| resume guard | `elastic.py prepare` | each candidate, newest first, goes through `scripts/check_ckpt_finite.py` under the run's python; a non-finite one is renamed `<name>.nonfinite` and the next older one is tried; if none is left the job fails (never a silent from-scratch start). Policy `check_ckpt_finite: false` opts out |
| halt | controller | a `NONFINITE` marker naming a tracked job halts the run, cancels its queued/running jobs, and sends `HALTED: NON-FINITE ...`. No requeue. `elastic.py resume <run>` clears the marker (look first) |

`scripts/check_ckpt_finite.py <ckpt>...` exits 0 only if every file is finite (3 = non-finite,
4 = unreadable). Test-only: `ELASTIC_TEST_NAN_AT_STEP=N` in a run's `env` poisons one trainable
parameter at global step N. It is unset by default and never belongs in a real run.

## Other commands

`pause <run> [--save-stop]`, `resume <run>`, `simulate-preempt <run>` (USR1, wait for the save,
`scontrol requeue`: what a real preemption does), `pick-ckpt <dir>`.
