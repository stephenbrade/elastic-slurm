# elastic-slurm

Preemption-safe, resizable training on Slurm. A controller keeps each run alive across preemptions, node
failures and stalls. It races a run over many GPU shapes and partitions at a fixed global batch, keeps the first
job that starts, cancels the rest, and grows the run to a bigger shape when one frees up. It also runs
bulk "pool" jobs (sharded encodes, eval renders) across many preemptible workers.

Extracted from a working training repo, where it runs on MIT CSAIL (and IBM Blue Vela). The detailed
design notes are in [`elastic/README.md`](elastic/README.md).

## What's here

| Path | What it is |
| --- | --- |
| `elastic/elastic.py` | Controller + CLI: `start`, `status`, `pause [--save-stop]`, `resume`, `controller`, `watchdog`, `simulate-preempt`, `fit`, … (stdlib only) |
| `elastic/job.sbatch` | The per-job launcher: lock, NaN/finite pre-checks, resume-checkpoint pick, signal forwarding |
| `elastic/job_pool.sbatch` | Worker for pool runs (claims shards, idempotent) |
| `elastic/job_liveband.sbatch` | Launcher variant for a plain (non-Lightning) trainer |
| `elastic/controller.sh` | Starts the controller in a loop (run it in tmux on the login node) |
| `elastic/watchdog.sbatch` | CPU job that restarts the controller if its heartbeat goes stale |
| `elastic/smoke_train.py` | Tiny DDP trainer for testing shapes, races and preemption end to end |
| `elastic/sites/*.json` | Per-cluster config: run root, python, CPU-job flags, named GPU *slots* (partition + account + QoS + gres) |
| `training/elastic_callback.py` | The in-trainer half (PyTorch Lightning): `ElasticCallback` (USR1 = save and keep going, USR2 = save and stop, time-based saves, non-finite-loss halt) and `slurm_topology()` (cluster env + atomic checkpoint IO). In the source repo it lives at `stable_audio_3/training/elastic.py`. |

## Setting it up on CSAIL

1. Copy `elastic/sites/csail.example.json` to `elastic/sites/csail.json`.
   - Replace `/data/hai-res/<you>` with your own directories (`root` holds the run dirs; `python` is a stdlib Python 3.11 for the controller).
   - Set `"mail"` to your address, or leave it out for no mail. `$ELASTIC_MAIL` also works.
   - Check the `slots` (accounts, partitions, QoS, gres names) against `sacctmgr show assoc user=$USER` for your account.
2. Edit the paths in `elastic/controller.sh` and `elastic/watchdog.sbatch`. They default to the original author's Blue Vela paths; on CSAIL set `ELASTIC_SITE=csail`, `ELASTIC_HOME` and `ELASTIC_PY`.
3. Make your trainer use the callback. Add `ElasticCallback` to the Lightning `Trainer` and take `plugins`/`cluster_environment` from `slurm_topology()`. Resume from the checkpoint path the launcher passes in.
4. Write a run: `<root>/<name>/run.json` with `script`, `args`, `python`, `repo`, `global_bs`, `micro_bs` and `configs` (a list of `[nodes, gpus_per_node, "<slot>"]` shapes). See `elastic/README.md`.
5. `ELASTIC_SITE=csail python3 -I elastic/elastic.py start <name>`, and keep `controller.sh` running in tmux.

## CSAIL gotchas we hit

- `sbatch --export=ALL,VAR=value` jobs are silently "CANCELLED by 0" within seconds, with empty logs. The CSAIL site file sets `export_via_env`, so elastic passes variables another way. Do the same in any sbatch you write by hand.
- The `lab-free` QoS on the shared partitions has GraceTime 0, so a preemption kills the job immediately. Keep `policy.save_every_s` short (we use 300 s).
- Multi-node over RoCE only works inside one 128.30.x /24. That's what the slots' `multinode_constraint` / `multinode_exclude` are for. Multi-node with real trainers was still being verified as of 2026-11-18.
- `hai-res-low` is a preemptible QoS on hai-res-l40s with no per-user GPU cap. It's the `hai-l40s-low` slot in the example site file.
