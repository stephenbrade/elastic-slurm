#!/usr/bin/env python3
"""Elastic, preemption-tolerant training runs on Blue Vela (Slurm). Stdlib only, Python >= 3.9.

A *run* is a directory ``$ELASTIC_ROOT/<name>/`` (default root ``/proj/spans/brade/lmdm/runs``):

    run.json       what to train: script + args, global batch, micro batch, allowed GPU configs
    state.json     controller-owned: the live primary job, any upgrade probe, failure count
    events.jsonl   append-only log of everything that happened (jobs append here too)
    ckpt/          --save_dir for the trainer (rolling, milestone and elastic-step=N.ckpt files)
    logs/          one Slurm log per job id

Every job launched by ``job.sbatch`` sets ``--batch_size <micro>`` and
``--accum_batches global/(micro x nodes x gpus)``, then resumes with ``--ckpt_path`` from the newest
valid checkpoint. So the effective batch is the same on any allowed config.

The controller (``elastic.py controller``, run in tmux on login1: cron and scrontab are disabled
here) polls every run each ``--interval`` seconds and:

  * RACES shapes (2026-10-27): whenever a run needs GPUs (first start, a preemption requeue, a
    crash) one job is queued per allowed config, and the first to start takes the run's start lock
    (``<run>/.train_lock``, in job.sbatch); any other starter exits with ``lost_race`` and the next
    tick cancels the rest. A queued job is never cancelled to try another shape, so a requeue keeps
    its queue age. No capacity or start-time estimates are used: on Blue Vela they all overpromised.
  * scales UP: when the primary has run for ``fit_dwell_s`` on fewer GPUs than a config allows and
    the remaining time (measured rate x ``--steps``) makes the move pay by more than
    ``switch_cost_s + switch_min_gain_s``, it queues a *probe* at the bigger shape. When the probe
    actually starts it sends the old primary SIGUSR2 (save + exit), waits for it to leave, and
    resumes from that save. The small allocation is never given up before the big one is held.
  * resubmits after TIMEOUT and, up to ``max_failures`` times, after a crash; marks the run done on
    a clean finish; treats a CANCELLED primary it did not cancel itself as the user pausing the run.

Notifications are Slurm mail, the only mail path that works here (login-node sendmail has no
running postfix): preempted/failed/finished jobs mail via their own ``--mail-type``, and the
controller announces its own actions by submitting a 1-second cpu job whose *name* is the message.

    elastic.py start   <run>  [--config I]     validate run.json, submit the first primary
    elastic.py status  [<run>]
    elastic.py pause   <run>  [--save-stop]    controller stops managing it (optionally save+exit)
    elastic.py resume  <run>                   back under management (submits if nothing is live)
    elastic.py controller [--interval S] [--once] [--role main|watchdog]
    elastic.py adopt   <run> <jobid> [--dry-run]   (kind=liveband) take over a live plain-sbatch job
    elastic.py watchdog                         (inside watchdog.sbatch) tick for a dead controller
    elastic.py simulate-preempt <run>          USR1 (save) then scontrol requeue, as a real preempt
    elastic.py prepare <run>                   (inside a job) print the exports job.sbatch evals
    elastic.py pick-ckpt <dir>
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import zipfile

HOME = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- site (2026-11-15)
# Everything cluster-specific comes from elastic/sites/<ELASTIC_SITE>.json (default "bluevela", whose
# values are exactly the old hard-codes, so Blue Vela behaviour is unchanged). CSAIL (sites/csail.json)
# adds named GPU *slots* (partition + account + QoS + gres type): a shape [n, g, "<slot>"] or a pool
# worker on "<slot>" is submitted to that slot's partition instead of the run-wide one.
SITE_NAME = os.environ.get("ELASTIC_SITE", "bluevela")
SITE_DEFAULTS = {
    "tag": "bluevela",
    "root": "/proj/spans/brade/lmdm/runs",
    "python": "/usr/bin/python3",                 # stdlib python for the controller / watchdog
    "cpu_job": ["-A", "slurm_grp_preemptable", "-p", "cpu", "--qos=cpu"],
    "slurm_mail": True,
    "watchdog_log": None,                         # None: watchdog.sbatch's own -o
    "slurm": {},
    "slots": {},
}


def _load_site(name):
    site = json.loads(json.dumps(SITE_DEFAULTS))
    p = os.path.join(HOME, "sites", f"{name}.json")
    if os.path.isfile(p):
        with open(p) as f:
            site.update(json.load(f))
    elif name != "bluevela":
        raise SystemExit(f"ELASTIC_SITE={name}: no {p}")
    return site


SITE = _load_site(SITE_NAME)


def mail_to(run):
    """Notification address: the run's "mail", else the site's "mail", else $ELASTIC_MAIL; "" = no mail."""
    return run.get("mail") or SITE.get("mail") or os.environ.get("ELASTIC_MAIL", "")
SITE_ENV_PREFIX = "" if SITE_NAME == "bluevela" else f"ELASTIC_SITE={SITE_NAME} "
ROOT = os.environ.get("ELASTIC_ROOT", SITE["root"])

POLICY_DEFAULTS = {
    "fit_dwell_s": 600,           # running this long before a move up is considered
    "probe_patience_s": 900,      # how long an upgrade probe may sit pending before it is dropped
    "probe_backoff_s": 1800,      # after a failed probe; doubles up to probe_backoff_max_s
    "probe_backoff_max_s": 7200,
    "stall_s": 600,               # a RUNNING job whose log has not changed this long is hung
                                  # (2x4 NCCL hangs at step 0, 2026-10-25; a crashed-but-alive job,
                                  # 2026-10-27): cancel it and race again. A training job's tqdm
                                  # line updates every few seconds, so 10 min of silence is a hang
    "max_stalls": 10,             # stalls are infrastructure (2026-10-28: three 2-node startups
                                  # hung on NCCL NET/IB status=12) and are counted apart from
                                  # crashes; only this many IN A ROW halt the run (the count
                                  # restarts at every new checkpoint and on `resume`)
    "max_failures": 2,            # crash resubmits before the run is halted
    "handoff_timeout_s": 900,     # probe waits this long for the old primary to save and exit
    "gpu_preference": ["h100", "h200"],   # probe order only (which type to try first)
    # Moving a RUNNING job to more GPUs (always via a held probe allocation):
    "switch_cost_s": 900,         # startup + handoff + resume overhead of one switch
    "switch_min_gain_s": 600,     # and the switch must save at least this much on top
    "scale_efficiency": 0.9,      # throughput of w_new GPUs ~= w_new / w_cur * this
    "assume_remaining_s": 21600,  # remaining time when the run's rate/steps are not yet known
    # Checkpointing inside the job (ElasticCallback, exported by `prepare`):
    "save_every_s": 600,          # rolling elastic-step save at least this often
    "grace_save_every_s": 240,    # after the preemption signal (start of the 10 min grace)
    # NaN guard (2026-11-12): `prepare` loads each resume candidate with the run's own python
    # (scripts/check_ckpt_finite.py) and skips + quarantines any whose weights are non-finite.
    "check_ckpt_finite": True,
    "ckpt_check_timeout_s": 900,
    # kind "liveband" only (2026-11-13): progress = new `step` records in <out_dir>/metrics.jsonl
    # (every 50 steps, ~6 s). Before the first record of a job, this longer limit applies.
    "startup_stall_s": 1800,
    # Shape backoff (2026-11-17, CSAIL; off on Blue Vela): a job that FAILS without any checkpoint
    # progress (NCCL init on a broken node pair, a node whose GPUs CUDA cannot see) is charged to its
    # SHAPE, not to the run: after shape_fail_max such failures in a row that shape sits out
    # shape_backoff_s (doubling to shape_backoff_max_s) while the others keep racing. Before this, on
    # CSAIL 2x2 vs-a100 ("Could not find NET with id 0") started within a minute, crashed and was
    # re-raced every tick forever, and a primary crash of that kind halted the whole run after
    # max_failures. max_noprogress_failures no-progress failures in a row (any shape) still halt it:
    # a trainer that crashes everywhere is not a shape problem.
    "shape_backoff": False,
    "shape_fail_max": 2,
    "shape_backoff_s": 3600,
    "shape_backoff_max_s": 6 * 3600,
    "max_noprogress_failures": 8,
}
POLICY_DEFAULTS.update(SITE.get("policy", {}))
NONFINITE_MARKER = "NONFINITE"    # <run>/NONFINITE, written by the trainer or `prepare` on a NaN stop
CHECKER = os.path.join("scripts", "check_ckpt_finite.py")   # relative to run["repo"]
SLURM_DEFAULTS = {
    "account": "slurm_grp_preemptable", "partition": "gpu-low", "qos": "burst",
    "constraint": "h200", "cpus_per_gpu": 8, "mem_per_gpu_gb": 96, "time": "7-00:00:00",
    "exclude": "", "signal_lead_s": 600,
}
SLURM_DEFAULTS.update(SITE.get("slurm", {}))
NOTIFY_MAX_PER_HOUR = 20

PENDINGISH = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "CONFIGURING", "SUSPENDED",
              "RESIZING"}
RUNNINGISH = {"RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING"}
# Only these end a tracked job. Anything else (UNKNOWN from a scontrol/sacct hiccup, a state this file
# does not list) keeps the job tracked: 2026-10-28, a probe whose lookup returned an odd state was
# dropped without a cancel, started 70 min later, took the start lock and trained untracked while
# the controller re-raced ~150 jobs into that lock.
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL",
            "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}


# ----------------------------------------------------------------------------- small utils

def now():
    return time.time()


def iso(t=None):
    return dt.datetime.fromtimestamp(t or now()).strftime("%Y-%m-%dT%H:%M:%S")


def sh(cmd, check=True):
    env = None
    if SITE.get("export_via_env") and cmd and cmd[0] == "sbatch":
        # CSAIL (2026-11-15): any --export other than plain ALL (e.g. ALL,K=V) makes the job fail at launch
        # with "user env retrieval failed requeued held" (tested: --export=ALL ok, ALL,FOO=1 and FOO=1 held).
        # So pass K=V through sbatch's own environment and keep --export=ALL.
        env, out = dict(os.environ), []
        for x in cmd:
            if x.startswith("--export=ALL,"):
                for kv in x[len("--export=ALL,"):].split(","):
                    k, _, v = kv.partition("=")
                    env[k] = v
                x = "--export=ALL"
            out.append(x)
        cmd = out
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> rc {r.returncode}: {r.stderr.strip()}")
    return r


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_dir(name_or_path):
    return name_or_path if os.sep in name_or_path else os.path.join(ROOT, name_or_path)


def event(rd, kind, **kw):
    rec = {"t": iso(), "event": kind, **kw}
    with open(os.path.join(rd, "events.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    # stderr: `prepare` prints shell exports on stdout that job.sbatch evals
    print(f"[{rec['t']}] {os.path.basename(rd)}: {kind} {kw if kw else ''}", file=sys.stderr,
          flush=True)


def kind(run):
    """'lightning' (scripts/train_lmdm.py, train_arc.py: job.sbatch) or 'liveband'
    (~/liveband_repro sbatch/train_data_ddp.sh: job_liveband.sbatch). 2026-11-13: the LiveBand
    runs were plain sbatch jobs; a requeue that failed instantly (0:53) left them dead for 8 h."""
    return run.get("kind", "lightning")


def lb_progress(run):
    """(step, mtime) of the newest train record in <out_dir>/metrics.jsonl (LiveBand rank 0 appends
    one every --log-every steps), or (-1, mtime or 0). Reads the last 64 KiB only."""
    p = os.path.join(run["out_dir"], "metrics.jsonl")
    try:
        mt = os.path.getmtime(p)
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return -1, 0.0
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and "val" not in r and isinstance(r.get("step"), int):
            return r["step"], mt
    return -1, mt


def progress_step(run, rd):
    """Lightning: newest loadable checkpoint step. LiveBand: newest logged train step."""
    if kind(run) == "liveband":
        return lb_progress(run)[0]
    return pick_ckpt(os.path.join(rd, "ckpt"))[1]


def lb_nonfinite_files(run):
    """The LiveBand trainer saves <out>/nonfinite_step<N>.pt and raises on a non-finite loss."""
    try:
        return sorted(f for f in os.listdir(run["out_dir"])
                      if f.startswith("nonfinite_step") and f.endswith(".pt"))
    except OSError:
        return []


def job_log(rd, p):
    """Adopted plain-sbatch jobs keep their own log path; elastic jobs log to <rd>/logs/<job>.out."""
    return p.get("log") or os.path.join(rd, "logs", f"{p['job']}.out")


def load_run(rd):
    run = read_json(os.path.join(rd, "run.json"))
    if run is None:
        raise SystemExit(f"no run.json in {rd}")
    run.setdefault("name", os.path.basename(rd))
    run["policy"] = {**POLICY_DEFAULTS, **run.get("policy", {})}
    run["slurm"] = {**SLURM_DEFAULTS, **run.get("slurm", {})}
    return run


def cfg_str(cfg):
    """'1x8gpu', or '1x8gpu-h100' for a shape that names its own GPU type (configs entry [n, g, type])."""
    n, g = cfg[:2]
    return f"{n}x{g}gpu" + (f"-{cfg[2]}" if len(cfg) > 2 else "")


def cfg_slot(run, cfg):
    """The site slot a shape names as its third element (CSAIL: partition/account/QoS/gres), else None."""
    if len(cfg) > 2:
        slots = {**SITE.get("slots", {}), **run.get("slots_def", {})}
        return slots.get(cfg[2])
    return None


def cfg_constraint(run, cfg):
    """A shape may pin its GPU type as a third element; otherwise the run-wide slurm constraint."""
    return cfg[2] if len(cfg) > 2 else run["slurm"].get("constraint")


def batch_for(run, cfg):
    """(micro_bs, accum) for a shape, or None if the global batch cannot be split evenly over it.

    With ``max_micro_bs`` in run.json (2026-10-23): the per-GPU share global/world is used as ONE batch
    when it fits under the cap, and accumulation appears only beyond the cap (largest divisor of the
    share that fits). Small shapes then run fuller GPUs instead of accumulating tiny batches; measured on
    the 88/4 student, 1x4 at micro 32 ran 2.44 steps/s against ~1.1 at micro 16 x accum 2.
    Without ``max_micro_bs`` the fixed ``micro_bs`` and the old accumulation rule apply unchanged.
    """
    world = cfg[0] * cfg[1]
    if run["global_bs"] % world:
        return None
    share = run["global_bs"] // world
    if "max_micro_bs" not in run:
        m = run["micro_bs"]
        return (m, share // m) if share % m == 0 else None
    cap = int(run["max_micro_bs"])
    m = max(d for d in range(1, min(cap, share) + 1) if share % d == 0)
    return m, share // m


def accum_for(run, cfg):
    b = batch_for(run, cfg)
    return b[1] if b else None


def validate_liveband(run):
    errs = [f"run.json missing '{k}'" for k in ("script", "args", "configs", "python", "repo",
                                                "out_dir", "env") if k not in run]
    if errs:
        return errs
    if not run["env"].get("RUN"):
        errs.append("env.RUN (the LiveBand run name) is required")
    elif os.path.basename(os.path.normpath(run["out_dir"])) != run["env"]["RUN"]:
        errs.append(f"out_dir {run['out_dir']} does not end in env.RUN {run['env']['RUN']}")
    if not run["env"].get("DATA"):
        errs.append("env.DATA is required (train_data_ddp.sh would silently default to slakh_sames)")
    batch = 128
    if "--batch" in run["args"]:
        batch = int(run["args"][run["args"].index("--batch") + 1])
    for cfg in run["configs"]:
        if cfg[0] != 1:
            errs.append(f"config {cfg_str(cfg)}: train_data_ddp.sh is single-node only")
        elif cfg[1] < 1 or batch % cfg[1]:
            errs.append(f"config {cfg_str(cfg)}: {cfg[1]} GPUs do not divide --batch {batch}")
    for bad in ("--out", "--name", "--data", "--direction"):
        if bad in run["args"]:
            errs.append(f"args must not contain {bad}: train_data_ddp.sh sets it (use env)")
    if max_steps_of(run) is None:
        errs.append("args need --steps (the controller stops the run there)")
    return errs


def validate(run):
    if kind(run) == "liveband":
        return validate_liveband(run)
    if kind(run) == "pool":
        errs = [f"run.json missing '{k}'" for k in ("cmd", "done", "shards", "slots", "python", "repo") if k not in run]
        if errs:
            return errs
        slots = {**SITE.get("slots", {}), **run.get("slots_def", {})}
        errs += [f"unknown slot '{k}' (site {SITE_NAME})" for k in run["slots"] if k not in slots]
        ids = [x["id"] for x in run["shards"]]
        if len(set(ids)) != len(ids):
            errs.append("duplicate shard ids")
        for x in run["shards"][:1]:
            try:
                pool_done_path(run, x)
                [c.format(**x["vars"]) for c in run["cmd"]]
            except (KeyError, IndexError, ValueError) as e:
                errs.append(f"shard {x['id']}: template error {e}")
        return errs
    errs = []
    for k in ("script", "args", "global_bs", "micro_bs", "configs", "python", "repo"):
        if k not in run:
            errs.append(f"run.json missing '{k}'")
    if errs:
        return errs
    for cfg in run["configs"]:
        if accum_for(run, cfg) is None:
            errs.append(f"config {cfg_str(cfg)}: global_bs {run['global_bs']} not divisible by "
                        f"micro {run['micro_bs']} x {cfg[0] * cfg[1]} GPUs")
    for bad in ("--batch_size", "--accum_batches", "--save_dir", "--ckpt_path"):
        if bad in run["args"]:
            errs.append(f"args must not contain {bad}: the launcher sets it")
    return errs


# ----------------------------------------------------------------------------- checkpoints

def ckpt_candidates(d):
    """[(step, path)] newest first, of files that LOAD as zips. A zip whose central directory reads
    is complete (torch.save writes it last), so a save killed mid-write is skipped. Only names
    ending in ``step=N.ckpt`` count: an in-flight ``*.ckpt.tmp`` (atomic saves, 2026-11-12) and a
    quarantined ``*.ckpt.nonfinite`` never match."""
    if not os.path.isdir(d):
        return []
    cands = []
    for f in os.listdir(d):
        m = re.search(r"step=(\d+)\.ckpt$", f)
        if m:
            p = os.path.join(d, f)
            try:
                cands.append((int(m.group(1)), os.path.getmtime(p), p))
            except OSError:            # vanished between listdir and stat (pruned, renamed)
                continue
    out = []
    for step, _, p in sorted(cands, reverse=True):
        try:
            with zipfile.ZipFile(p) as z:
                z.namelist()
            out.append((step, p))
        except Exception:
            continue
    return out


def pick_ckpt(d):
    """Newest *loadable* checkpoint by step (zip test only: this runs in the stdlib controller).
    ``prepare`` additionally requires finite weights, see ``pick_finite_ckpt``."""
    for step, p in ckpt_candidates(d):
        return p, step
    return None, -1


def check_finite(run, path):
    """('finite'|'nonfinite'|'unreadable'|'unknown', detail) for one checkpoint, by running
    scripts/check_ckpt_finite.py under the run's own python (this file stays stdlib-only).
    'unknown' = the check could not run (no checker in that repo, crash, timeout)."""
    # A LiveBand run's repo has no checker: use this SA3 checkout (elastic/ lives in it).
    crepo = run.get("checker_repo") or (os.path.dirname(HOME) if kind(run) == "liveband" else run["repo"])
    checker = os.path.join(crepo, CHECKER)
    if not os.path.isfile(checker):
        return "unknown", f"no {checker}"
    try:
        r = subprocess.run([run["python"], checker, path], cwd=crepo, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True,
                           timeout=run["policy"].get("ckpt_check_timeout_s", 900))
    except subprocess.TimeoutExpired:
        return "unknown", "checker timed out"
    except OSError as e:
        return "unknown", f"{type(e).__name__}: {e}"
    out = (r.stdout.strip().splitlines() or [""])[-1][:400]
    status = {0: "finite", 3: "nonfinite", 4: "unreadable"}.get(r.returncode, "unknown")
    if status == "unknown":
        out = f"checker rc {r.returncode}: {(r.stderr.strip().splitlines() or [''])[-1][:300]}"
    return status, out


def pick_finite_ckpt(run, rd, d):
    """(path, step, skipped) for ``prepare``: the newest zip-valid checkpoint whose weights are
    finite. A non-finite one is renamed ``<name>.nonfinite`` (out of every resume/prune pattern,
    kept for forensics) and the next older is tried; an unreadable one is skipped. ``skipped`` is
    [(basename, status, detail)]. 2026-11-12: ss_bc1_lag4_jam_mix2stem_cenc resumed from a NaN
    checkpoint and trained NaN for 78k steps, because this test used to be the zip test alone."""
    skipped = []
    for step, p in ckpt_candidates(d):
        if not run["policy"].get("check_ckpt_finite", True):
            return p, step, skipped
        status, detail = check_finite(run, p)
        if status == "finite":
            event(rd, "ckpt_finite", ckpt=os.path.basename(p), detail=detail)
            return p, step, skipped
        if status == "unknown":
            # Cannot tell (old repo without the checker, broken env): resume as before, loudly.
            event(rd, "ckpt_check_unavailable", ckpt=os.path.basename(p), detail=detail)
            return p, step, skipped
        skipped.append((os.path.basename(p), status, detail))
        if status == "nonfinite":
            q = p + ".nonfinite"
            try:
                os.replace(p, q)
            except OSError as e:
                detail += f" (quarantine rename failed: {e})"
        event(rd, "ckpt_skipped", ckpt=os.path.basename(p), status=status, detail=detail)
    return None, -1, skipped


def nonfinite_marker(rd):
    return read_json(os.path.join(rd, NONFINITE_MARKER))


def clear_nonfinite_marker(rd):
    p = os.path.join(rd, NONFINITE_MARKER)
    if os.path.exists(p):
        os.replace(p, f"{p}.cleared.{int(now())}")
        event(rd, "nonfinite_marker_cleared")


def prune_ckpts(rd, keep=2, min_age_s=600, dry_run=False):
    """Delete stale checkpoints in <rd>/ckpt (2026-10-23, user request).

    Lightning's ``--checkpoint-keep 1`` only prunes within one job, so every restart left its last
    checkpoint behind (26-42 GB per run after a few hours). Kept, always:
      * every ``milestone-*`` file (the permanent --checkpoint-milestone-every copies);
      * the ``keep`` newest regular checkpoints that LOAD (zip central directory reads, the same test
        ``pick_ckpt`` resumes with), so a corrupt newest file never leaves the run without a good one;
      * anything modified in the last ``min_age_s`` (could be mid-write);
      * whatever ``pick_ckpt`` would resume from right now.
    Returns [(path, bytes)] deleted (or that would be, with dry_run).
    """
    d = os.path.join(rd, "ckpt")
    if not os.path.isdir(d):
        return []
    regular = []
    for f in os.listdir(d):
        m = re.search(r"step=(\d+)\.ckpt$", f)
        if m and not f.startswith("milestone-"):
            regular.append((int(m.group(1)), os.path.join(d, f)))
    regular.sort(reverse=True)
    keepers, good = set(), 0
    for step, p in regular:
        if good >= keep:
            break
        keepers.add(p)
        try:
            with zipfile.ZipFile(p) as z:
                z.namelist()
            good += 1
        except Exception:
            continue                      # unreadable: kept for now (could be mid-write), not counted
    resume, _ = pick_ckpt(d)
    if resume:
        keepers.add(resume)
    gone = []
    for _, p in regular:
        if p in keepers or now() - os.path.getmtime(p) < min_age_s:
            continue
        size = os.path.getsize(p)
        if not dry_run:
            os.remove(p)
        gone.append((p, size))
    # Atomic saves (2026-11-12) stage as <name>.ckpt.tmp; one left by a save killed mid-write is
    # dead weight. Never a resume candidate; removed once it is clearly not in flight (6 h).
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            if f.endswith(".ckpt.tmp") and now() - os.path.getmtime(p) > 6 * 3600:
                size = os.path.getsize(p)
                if not dry_run:
                    os.remove(p)
                gone.append((p, size))
        except OSError:
            continue
    return gone


# ----------------------------------------------------------------------------- slurm

def slot_cpus_mem(run, cfg):
    """(cpus_per_gpu, mem_per_gpu_gb) for a shape: the run's values, capped by the slot's
    ``max_cpus_per_gpu`` / ``max_mem_per_gpu_gb`` (CSAIL 2026-11-17: V100 / 2080 Ti / Titan RTX nodes
    have 80 CPUs and ~500 GB for 8 GPUs, so 12 CPUs + 64 GB per GPU made every 1x8 there unsatisfiable)."""
    s = run["slurm"]
    slot = cfg_slot(run, cfg) or {}
    c, m = s["cpus_per_gpu"], s["mem_per_gpu_gb"]
    if slot.get("max_cpus_per_gpu"):
        c = min(c, slot["max_cpus_per_gpu"])
    if slot.get("max_mem_per_gpu_gb"):
        m = min(m, slot["max_mem_per_gpu_gb"])
    return c, m


def sbatch_flags(run, rd, cfg, role, primary_job=None):
    s = run["slurm"]
    n, g = cfg[:2]
    slot = cfg_slot(run, cfg)
    acct, part, qos = (slot["account"], slot["partition"], slot["qos"]) if slot else (s["account"], s["partition"], s["qos"])
    gres = f"--gres=gpu:{slot['gres']}:{g}" if slot else f"--gres=gpu:{g}"
    cpg, mpg = slot_cpus_mem(run, cfg)
    flags = [
        "-A", acct, "-p", part, f"--qos={qos}",
        "-N", str(n)] + ([] if s.get("no_gres") else [gres]) + [
        f"--ntasks-per-node={g if n > 1 else 1}",
        "-c", str(cpg * (1 if n > 1 else g)),
        f"--mem={mpg * g}G", "-t", s["time"],
        "--requeue", "--open-mode=append", f"--signal=B:USR1@{s['signal_lead_s']}",
    ] + ([f"--mail-type=REQUEUE,FAIL,END", f"--mail-user={mail_to(run)}"] if mail_to(run) else []) + [
        "-J", f"el:{run['name']}:{cfg_str(cfg)}{':probe' if role == 'probe' else ''}",
        "-o", os.path.join(rd, "logs", "%j.out"),
        f"--export=ALL,ELASTIC_RUN={rd},ELASTIC_HOME={HOME},ELASTIC_ROLE={role},"
        f"ELASTIC_PRIMARY={primary_job or ''},ELASTIC_SITE={SITE_NAME},ELASTIC_PY={SITE['python']}",
    ]
    if not SITE["slurm_mail"]:
        flags = [x for x in flags if not x.startswith("--mail-")]
    if n > 1 and s.get("exclusive_multinode", True):
        flags.append("--exclusive")
    if cfg_constraint(run, cfg) and not slot:
        flags.append(f"--constraint={cfg_constraint(run, cfg)}")
    # A multi-node shape on a site whose nodes sit in several buildings (CSAIL: holyoke / stata) asks
    # for all nodes in ONE of them: Slurm's "[a|b]" matching-OR constraint.
    mn = s.get("multinode_constraint") if n > 1 else None
    if slot and slot.get("constraint"):
        flags.append(f"--constraint={slot['constraint']}")
    elif mn:
        flags.append(f"--constraint={mn}")
    ex = [x for x in (s.get("exclude") or "").split(",") if x]
    if n > 1 and slot:
        # CSAIL 2026-11-17: RoCE (NCCL NET/IB) works only between nodes on the same /24 subnet, and
        # sockets between them ran at 0.16 GB/s, so multi-node shapes avoid the nodes listed here.
        ex += [x for x in slot.get("multinode_exclude", []) if x not in ex]
    if ex:
        flags.append(f"--exclude={','.join(ex)}")
    return flags


def submit(run, rd, cfg_idx, role, primary_job=None):
    cfg = run["configs"][cfg_idx]
    script = "job_liveband.sbatch" if kind(run) == "liveband" else "job.sbatch"
    r = sh(["sbatch", "--parsable"] + sbatch_flags(run, rd, cfg, role, primary_job)
           + [os.path.join(HOME, script)])
    jid = r.stdout.strip().split(";")[0]
    mb = batch_for(run, cfg) if kind(run) != "liveband" else None
    event(rd, "submitted", job=jid, role=role, config=cfg_str(cfg), micro=mb and mb[0], accum=mb and mb[1])
    return {"job": jid, "config": cfg_idx, "submitted": now(), "restarts": 0,
            "pending_since": now(), "started": None}


# ----------------------------------------------------------------------------- capacity (2026-10-23)
# Pick shapes from what is free NOW instead of only from how long a job has pended. One
# `scontrol show node` per tick. "Free" is a hint, not a promise: on gpu-low (lowest tier) backfill can
# hold free resources for higher-priority jobs, so this only feeds the read-only `fit` hint.

def capacity(run):
    """[(node, free_gpus, free_cpus, free_mem_gb, features)] for schedulable nodes in this run's
    partition (and not excluded). Each shape filters by its own GPU type in ``fits``."""
    s = run["slurm"]
    r = sh(["scontrol", "show", "node", "-o"], check=False)
    excl = set(filter(None, (s.get("exclude") or "").split(",")))
    out = []
    for line in r.stdout.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S*)", line))
        node = kv.get("NodeName", "")
        if not node or node in excl:
            continue
        parts = {s["partition"]} | {sl["partition"] for sl in (cfg_slot(run, c) for c in run.get("configs", [])) if sl}
        if not parts & set(kv.get("Partitions", "").split(",")):
            continue
        feats = set(kv.get("AvailableFeatures", "").split(","))
        feats |= {f"part:{x}" for x in kv.get("Partitions", "").split(",")}   # slot shapes match on partition
        state = kv.get("State", "").upper()
        # PLANNED = backfill is holding the node for a higher-priority job: free, but not ours to get.
        if any(x in state for x in ("DOWN", "DRAIN", "FAIL", "MAINT", "RESERVED", "NOT_RESPONDING", "POWER", "PLANNED")):
            continue
        def gpus(tres):
            m = re.search(r"gres/gpu=(\d+)", tres or "")
            return int(m.group(1)) if m else 0
        fg = gpus(kv.get("CfgTRES")) - gpus(kv.get("AllocTRES"))
        fc = int(kv.get("CPUTot", 0) or 0) - int(kv.get("CPUAlloc", 0) or 0)
        fm = (int(kv.get("RealMemory", 0) or 0) - int(kv.get("AllocMem", 0) or 0)) // 1024
        if fg > 0:
            out.append((node, fg, fc, fm, feats))
    return out


def world(cfg):
    return cfg[0] * cfg[1]


def hosts(run, cfg, cap):
    """How many nodes in `cap` can each take g GPUs of this shape's type with the run's CPU/mem."""
    g = cfg[1]
    slot = cfg_slot(run, cfg)
    c = f"part:{slot['partition']}" if slot else cfg_constraint(run, cfg)
    cpg, mpg = slot_cpus_mem(run, cfg)
    return sum(1 for _, fg, fc, fm, feats in cap
               if (not c or c in feats)
               and fg >= g and fc >= g * cpg and fm >= g * mpg)


# ----------------------------------------------------------------------------- race (2026-10-27)
# Shape choice is Slurm's, not ours. Every estimate the controller had overpromised on Blue Vela
# (free-GPU snapshots held by backfill, --test-only floored at now+600 s, squeue --start hours off),
# and acting on them cost 22-48 shape moves per run in one evening, each giving up queue age. Now a
# run that needs GPUs queues ONE JOB PER ALLOWED SHAPE at once (the submit filter forbids naming two
# GPU models in one job) and the first to start wins: job.sbatch takes the run's start lock
# (<run>/.train_lock), a second starter exits with `lost_race`, and the next tick cancels the rest.
# A queued job is never cancelled to try another shape.

def max_steps_of(run):
    a = run.get("args", [])
    for flag in ("--steps", "--max_steps", "--max-steps"):
        if flag in a:
            try:
                return int(a[a.index(flag) + 1])
            except (ValueError, IndexError):
                return None
    return None


def remaining_seconds(run, rd, p):
    """Time left on the current shape from this job's own progress: (ckpt step - resume step) over
    (ckpt mtime - job_prepare time). Falls back to policy assume_remaining_s."""
    pol = run["policy"]
    total = max_steps_of(run)
    ck, step = pick_ckpt(os.path.join(rd, "ckpt"))
    if total is None or ck is None:
        return pol["assume_remaining_s"]
    prep = None
    try:
        with open(os.path.join(rd, "events.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("event") == "job_prepare" and str(r.get("job")) == str(p["job"]):
                    prep = r
    except (OSError, ValueError):
        pass
    if prep is None:
        return pol["assume_remaining_s"]
    t0 = time.mktime(time.strptime(prep["t"], "%Y-%m-%dT%H:%M:%S"))
    s0 = int(prep.get("resume_step", -1))
    dt_s = os.path.getmtime(ck) - t0
    if step <= max(s0, 0) or dt_s <= 0:
        return pol["assume_remaining_s"]
    rate = (step - max(s0, 0)) / dt_s
    return max(0.0, (total - step) / rate)


def switch_gain_s(run, rd, p, cur_cfg, new_cfg):
    """Seconds saved by moving the rest of the run from cur_cfg to new_cfg, net of switch_cost_s."""
    pol = run["policy"]
    t_rem = remaining_seconds(run, rd, p)
    speed = world(new_cfg) / world(cur_cfg) * pol["scale_efficiency"]
    if speed <= 1.0:
        return -pol["switch_cost_s"]
    return t_rem * (1.0 - 1.0 / speed) - pol["switch_cost_s"]


def job_info(jid):
    """{'state', 'reason', 'restarts', 'exit'} from scontrol, falling back to sacct once the
    controller has purged the record (MinJobAge)."""
    r = sh(["scontrol", "show", "job", "-o", str(jid)], check=False)
    if r.returncode == 0 and "JobState=" in r.stdout:
        kv = dict(re.findall(r"(\w+)=(\S*)", r.stdout))
        return {"state": kv.get("JobState", "UNKNOWN"), "reason": kv.get("Reason", ""),
                "restarts": int(kv.get("Restarts", "0") or 0), "exit": kv.get("ExitCode", "")}
    r = sh(["sacct", "-X", "-n", "-P", "-j", str(jid), "-o", "State,ExitCode,Reason"], check=False)
    line = (r.stdout.strip().splitlines() or [""])[-1]
    if not line:
        return {"state": "UNKNOWN", "reason": "", "restarts": 0, "exit": ""}
    parts = line.split("|") + ["", ""]
    return {"state": parts[0].split()[0], "reason": parts[2], "restarts": 0, "exit": parts[1]}


_notify_times = []


NTFY_TOPIC_FILE = os.path.join(os.path.dirname(ROOT), ".ntfy_topic")
SMTP_FILE = os.path.join(os.path.dirname(ROOT), ".smtp.json")


_NTFY_TOPIC = None


def _send_ntfy(title, body):
    """Phone push via ntfy.sh (HTTPS is the one egress that works here). The topic name is the
    only secret, so it lives in a 0600 file, never in the repo."""
    import urllib.request
    global _NTFY_TOPIC
    try:
        _NTFY_TOPIC = open(NTFY_TOPIC_FILE).read().strip() or _NTFY_TOPIC
    except OSError:
        pass                      # /proj gone: the watchdog still has the topic it read at start
    topic = _NTFY_TOPIC
    if not topic:
        return "no topic file"
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode(),
                                 headers={"Title": title[:200], "Priority": "high"})
    try:
        urllib.request.urlopen(req, timeout=20).read()
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]


def _send_smtp(to, subject, body):
    """Authenticated SMTP submission (587 STARTTLS is open; 25 is blocked). Credentials in a 0600
    JSON file: {"host", "port", "user", "password", "from"}."""
    import smtplib
    from email.message import EmailMessage
    cfg = read_json(SMTP_FILE)
    if not cfg:
        return "no smtp config"
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = cfg.get("from", cfg["user"]), to, subject
    m.set_content(body)
    try:
        with smtplib.SMTP(cfg["host"], int(cfg.get("port", 587)), timeout=30) as c:
            c.starttls()
            c.login(cfg["user"], cfg["password"])
            c.send_message(m)
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]


def notify(run, rd, msg, urgent=False):
    """Alert on every channel that is configured: ntfy push, SMTP mail, and Slurm mail as a last
    resort. Slurm mail is kept although on 2026-10-23 none of it was delivered: the cluster's
    MTA is down and outbound port 25 is blocked. ``urgent`` (halts, controller errors, LiveBand
    resubmits) bypasses the hourly cap: on 2026-11-13 routine SA3 messages had used it up."""
    global _notify_times
    _notify_times = [t for t in _notify_times if now() - t < 3600]
    if len(_notify_times) >= NOTIFY_MAX_PER_HOUR and not urgent:
        event(rd, "notify_suppressed", msg=msg)
        return
    _notify_times.append(now())
    title = f"[{SITE['tag']}] {run['name']}: {msg}"
    body = (f"{msg}\n\nrun: {rd}\ntime: {iso()}\n"
            f"status: {SITE_ENV_PREFIX}{SITE['python']} -I {HOME}/elastic.py status {run['name']}\n")
    res = {"ntfy": _send_ntfy(title, body),
           "smtp": _send_smtp(mail_to(run), title, body) if mail_to(run) else "no mail address"}
    if not SITE["slurm_mail"] or not mail_to(run):
        event(rd, "notified", msg=msg, **res)
        return
    name = re.sub(r"[^\w :>=.,()/+-]", "_", f"ELASTIC {run['name']}: {msg}")[:150]
    r = sh(["sbatch", "--parsable"] + SITE["cpu_job"] + ["-c", "1",
            "--mem=256M", "-t", "00:01:00", "-J", name, "--mail-type=END",
            f"--mail-user={mail_to(run)}", "-o", "/dev/null",
            "--wrap", "true"], check=False)
    res["slurm_mail_job"] = r.stdout.strip() or r.stderr.strip()[:200]
    event(rd, "notified", msg=msg, **res)


# ----------------------------------------------------------------------------- controller

def lock_holder(rd):
    try:
        return open(os.path.join(rd, ".train_lock", "job")).read().strip()
    except OSError:
        return None


def holder_job(run, jid):
    """Config index of `jid` if it is a RUNNING job of this run (named el:<run>:<shape>[:probe]), else None."""
    if job_info(jid)["state"] != "RUNNING":
        return None
    r = sh(["squeue", "-h", "-j", str(jid), "-o", "%j"], check=False)
    name = (r.stdout or "").strip()
    parts = name.split(":")
    if len(parts) < 3 or parts[0] != "el" or parts[1] != run["name"]:
        return None
    for j, cfg in enumerate(run["configs"]):
        if cfg_str(cfg) == parts[2]:
            return j
    return None


def lost_race(rd, jid):
    """True if job `jid` exited 0 without training: it lost the start lock, or its preflight found
    a bad node (job.sbatch). Such a COMPLETED job is re-raced, never read as a finished run."""
    try:
        with open(os.path.join(rd, "events.jsonl")) as f:
            for l in f:
                if '"lost_race"' in l or '"bad_node"' in l:
                    r = json.loads(l)
                    if r.get("event") in ("lost_race", "bad_node") and str(r.get("job")) == str(jid):
                        return True
    except OSError:
        pass
    return False


def job_made_progress(run, rd, jid):
    """True if the run's checkpoint step moved past the resume step this job started from
    (its job_prepare event). A job that never got to prepare made none."""
    rs = None
    try:
        with open(os.path.join(rd, "events.jsonl")) as f:
            for line in f:
                if '"job_prepare"' in line:
                    r = json.loads(line)
                    if r.get("event") == "job_prepare" and str(r.get("job")) == str(jid):
                        rs = int(r.get("resume_step", -1))
    except (OSError, ValueError):
        pass
    return rs is not None and progress_step(run, rd) > max(rs, 0)


def shape_failed(run, rd, st, cfg_idx, jid, state):
    """Charge a no-progress failure to the shape (policy shape_backoff). Returns True if the run
    must halt (max_noprogress_failures in a row)."""
    pol = run["policy"]
    key = cfg_str(run["configs"][cfg_idx])
    sf = st.setdefault("shape_fail", {})
    n = sf.get(key, 0) + 1
    sf[key] = n
    st["noprogress_failures"] = st.get("noprogress_failures", 0) + 1
    ev = {"config": key, "job": jid, "state": state, "n": n, "run_noprogress": st["noprogress_failures"]}
    if n >= pol["shape_fail_max"]:
        bad = st.setdefault("submit_failed", {})
        k = (bad.get(key) or {}).get("k", 0) + 1
        wait = min(pol["shape_backoff_s"] * 2 ** (k - 1), pol["shape_backoff_max_s"])
        bad[key] = {"n": n, "k": k, "err": f"{n} failures without progress (last {state}, job {jid})",
                    "retry_after": now() + wait}
        sf[key] = 0
        ev["backoff_s"] = wait
    event(rd, "shape_failed", **ev)
    return st["noprogress_failures"] >= pol["max_noprogress_failures"]


SUBMIT_RETRY_S = (600, 6 * 3600)   # backoff after a rejected sbatch: first retry, cap


def try_submit(run, rd, st, j, role, primary_job=None):
    """submit(), but a rejected sbatch (CSAIL 2026-11-17: a shape no node of the partition can hold,
    "Requested node configuration is not available"; a QoS that refuses the account) is recorded in
    st["submit_failed"][<shape>] with exponential backoff and returns None, instead of raising out of
    the tick. Before this, one bad shape in a 90-shape race aborted the tick after some jobs were
    already queued, and state.json, written only at the end of the tick, never recorded them."""
    bad = st.setdefault("submit_failed", {})
    key = cfg_str(run["configs"][j])
    b = bad.get(key)
    if b and now() < b["retry_after"]:
        return None
    try:
        r = submit(run, rd, j, role, primary_job)
    except RuntimeError as e:
        n = (b or {}).get("n", 0) + 1
        wait = min(SUBMIT_RETRY_S[0] * 2 ** (n - 1), SUBMIT_RETRY_S[1])
        bad[key] = {"n": n, "err": str(e)[-300:], "retry_after": now() + wait}
        event(rd, "submit_failed", config=key, role=role, n=n, retry_in_s=wait, err=str(e)[-300:])
        return None
    bad.pop(key, None)
    return r


def race_submit(run, rd, st, reason):
    """Queue a job for every allowed shape that has none queued for this run yet."""
    taken = {r["config"] for r in st.get("race", [])}
    if st.get("primary"):
        taken.add(st["primary"]["config"])
    new, failed = [], False
    for j in range(len(run["configs"])):
        if j not in taken:
            r = try_submit(run, rd, st, j, "primary")
            if r is None:
                failed = True       # backoff state changed or is pending: state.json must be written
                continue
            st.setdefault("race", []).append(r)
            new.append(f"{r['job']}:{cfg_str(run['configs'][j])}")
    if new:
        event(rd, "race", reason=reason, jobs=new)
    elif not st.get("race") and not st.get("primary") and st.get("submit_failed") \
            and now() - st.get("nothing_queued_alert", 0) > 3600:
        st["nothing_queued_alert"] = now()
        notify(run, rd, f"NOTHING QUEUED: every shape's sbatch was rejected "
                        f"({len(st['submit_failed'])} shapes), see events.jsonl submit_failed", urgent=True)
    return bool(new) or failed


def halt_nonfinite(run, rd, st, mk):
    """The trainer (or `prepare`) recorded a non-finite stop: halt, cancel everything queued or
    running for this run, notify. Never requeued: resuming would only train NaN again (2026-11-12)."""
    jobs = [c["job"] for c in ([st["primary"]] if st.get("primary") else []) + st.get("race", [])
            + st.get("probes", [])]
    log = os.path.join(rd, "logs", f"{mk.get('job')}.out")
    if st.get("primary") and str(st["primary"]["job"]) == str(mk.get("job")):
        log = job_log(rd, st["primary"])
    for j in jobs:
        if job_info(j)["state"] not in TERMINAL:
            sh(["scancel", j], check=False)
    st["status"] = "halted"
    st["primary"], st["race"], st["probes"] = None, [], []
    event(rd, "halted", reason="nonfinite", job=mk.get("job"), where=mk.get("where"),
          detail=mk.get("detail"), cancelled=jobs, log=log)
    notify(run, rd, f"HALTED: NON-FINITE ({mk.get('where')}) in job {mk.get('job')}: "
                    f"{str(mk.get('detail'))[:160]} | log {log}", urgent=True)


def drop_queued_on_stop(rd, st):
    """A run that just became done / halted / paused keeps no queued race or probe jobs (2026-11-17:
    s0_race FINISHED at 11:47 with 14 upgrade probes still queued; a late starter would have taken the
    start lock of a finished run). Running probes are cancelled too: they are not the primary."""
    if st.get("status") == "active":
        return False
    jobs = [c["job"] for c in st.get("race", []) + st.get("probes", [])
            if not st.get("primary") or c["job"] != st["primary"]["job"]]
    if not jobs:
        return False
    for j in jobs:
        sh(["scancel", j], check=False)
    st["race"], st["probes"] = [], []
    event(rd, "queued_jobs_cancelled", reason=st.get("status"), jobs=jobs)
    return True


def tick(rd):
    run = load_run(rd)
    sp = os.path.join(rd, "state.json")
    st = read_json(sp)
    if not st or st.get("status") != "active":
        return
    if kind(run) == "pool":
        tick_pool(run, rd, st)
        write_json(sp, st)
        return
    pol = run["policy"]
    cfgs = run["configs"]
    st.setdefault("race", [])
    changed = False

    # 1. upgrade probes in flight. Moving up is a race too: one queued probe per bigger shape that
    #    pays; the first to start hands over (USR2 save-and-exit of the primary, then the start
    #    lock), the rest are cancelled. (`probe` is the pre-race single-probe field.)
    old_single = st.pop("probe", None)
    st["probes"] = st.get("probes", []) + ([old_single] if old_single else [])
    if st["probes"]:
        pinfo = {q["job"]: job_info(q["job"]) for q in st["probes"]}
        up = [q for q in st["probes"] if pinfo[q["job"]]["state"] in RUNNINGISH]
        if up:
            holder = lock_holder(rd)
            pr = next((q for q in up if q["job"] == holder), up[0])
            old = st["primary"]
            st["primary"] = {**pr, "started": now(), "pending_since": None}
            for q in st["probes"]:
                if q["job"] != pr["job"] and pinfo[q["job"]]["state"] in PENDINGISH | RUNNINGISH:
                    sh(["scancel", q["job"]], check=False)
            st["probes"] = []
            st["last_change"] = now()
            st["probe_backoff_s"] = pol["probe_backoff_s"]
            event(rd, "scaled_up", job=pr["job"], config=cfg_str(cfgs[pr["config"]]),
                  replaced=old["job"] if old else None)
            # The replaced job can requeue itself after its USR2 save-and-exit (seen 2026-10-23 on
            # student_v1 job 341547, which went back to PENDING). Two trainers on one ckpt dir would
            # corrupt the run, so the old job is cancelled explicitly rather than trusted to be gone.
            if old and old.get("job") and old["job"] != pr["job"]:
                sh(["scancel", old["job"]], check=False)
                event(rd, "replaced_job_cancelled", job=old["job"])
            notify(run, rd, f"scaled UP to {cfg_str(cfgs[pr['config']])} (job {pr['job']})")
            changed = True
        else:
            alive = [q for q in st["probes"] if pinfo[q["job"]]["state"] not in TERMINAL]
            if alive and now() - min(q["submitted"] for q in alive) > pol["probe_patience_s"]:
                for q in alive:
                    sh(["scancel", q["job"]], check=False)
                b = min(st.get("probe_backoff_s", pol["probe_backoff_s"]) * 2,
                        pol["probe_backoff_max_s"])
                st["probe_backoff_s"] = b
                st["next_probe_after"] = now() + b
                event(rd, "probes_dropped", jobs=[q["job"] for q in alive],
                      waited_s=int(now() - min(q["submitted"] for q in alive)), backoff_s=b)
                alive = []
            if len(alive) != len(st["probes"]):
                st["probes"] = alive
                changed = True

    # 1b. self-heal: the start lock names the job that is really training. If that is one of this
    #     run's RUNNING jobs and the controller is not tracking it (an orphan), adopt it as the
    #     primary and drop everything else; otherwise every raced job would lose the lock forever.
    holder = lock_holder(rd)
    tracked = {c["job"] for c in ([st["primary"]] if st.get("primary") else []) + st["race"] + st["probes"]}
    if holder and holder not in tracked:
        hj = holder_job(run, holder)
        if hj is not None:
            for c in ([st["primary"]] if st.get("primary") else []) + st["race"] + st["probes"]:
                sh(["scancel", c["job"]], check=False)
            st["primary"] = {"job": holder, "config": hj, "submitted": now(), "restarts": 0,
                             "pending_since": None, "started": now()}
            st["race"], st["probes"] = [], []
            event(rd, "adopted_lock_holder", job=holder, config=cfg_str(cfgs[hj]))
            notify(run, rd, f"adopted untracked trainer {holder} ({cfg_str(cfgs[hj])}) that holds the start lock")
            changed = True

    # 2. resolve the race: whichever of {primary, race} is RUNNING is the primary; the lock holder
    #    wins if two started between ticks (the other has already exited on the lock).
    cands = ([st["primary"]] if st.get("primary") else []) + st["race"]
    infos = {c["job"]: job_info(c["job"]) for c in cands}
    running = [c for c in cands if infos[c["job"]]["state"] == "RUNNING"]
    if running:
        holder = lock_holder(rd)
        win = next((c for c in running if c["job"] == holder), running[0])
        losers = [c for c in cands if c["job"] != win["job"]]
        for c in losers:
            if infos[c["job"]]["state"] in PENDINGISH | RUNNINGISH:
                sh(["scancel", c["job"]], check=False)
        if losers:
            event(rd, "race_won", job=win["job"], config=cfg_str(cfgs[win["config"]]),
                  cancelled=[c["job"] for c in losers])
            changed = True
        st["primary"], st["race"] = win, []
    else:
        if kind(run) == "liveband" and not st.get("primary"):
            # A job that started AND ended between two ticks was never seen RUNNING, so it is still
            # in `race`. Judge it as the primary (finished / exited early / FAILED counts toward
            # max_failures / CANCELLED = user pause) instead of dropping it and re-racing: smoke
            # test 2026-11-13, a run that finished in 75 s was resubmitted. (SA3 runs keep the old
            # behaviour: with many shapes a fast shape-specific failure is re-raced, not counted.)
            ended = [c for c in st["race"] if infos[c["job"]]["state"] in TERMINAL
                     and not lost_race(rd, c["job"])]
            if ended:
                st["primary"] = {**ended[-1], "started": ended[-1].get("started") or now()}
                st["race"] = [c for c in st["race"] if c["job"] != ended[-1]["job"]]
                event(rd, "race_job_ended_unseen", job=ended[-1]["job"],
                      state=infos[ended[-1]["job"]]["state"])
                changed = True
        if kind(run) != "liveband" and pol["shape_backoff"]:
            for c in st["race"]:
                stt = infos[c["job"]]["state"]
                if stt in TERMINAL and stt not in ("COMPLETED", "CANCELLED", "PREEMPTED", "TIMEOUT") \
                        and not lost_race(rd, c["job"]) and not job_made_progress(run, rd, c["job"]):
                    if shape_failed(run, rd, st, c["config"], c["job"], stt):
                        st["status"] = "halted"
                        event(rd, "halted", state="NOPROGRESS_FAILURES", n=st["noprogress_failures"])
                        notify(run, rd, f"HALTED: {st['noprogress_failures']} failures in a row without "
                                        f"progress on any shape (last {cfg_str(cfgs[c['config']])} job {c['job']})",
                               urgent=True)
                        write_json(sp, st)
                        return
        alive = [c for c in st["race"] if infos[c["job"]]["state"] not in TERMINAL]
        if len(alive) != len(st["race"]):
            st["race"] = alive
            changed = True

    # NaN guard (2026-11-12): a tracked job that stopped on non-finite values halts the run, whatever
    # state Slurm reports for it (FAILED, COMPLETED, or still RUNNING while a rank hangs on exit).
    if kind(run) == "liveband" and st.get("primary"):
        # The LiveBand trainer saves nonfinite_step<N>.pt and raises. A new one halts the run
        # (CLAUDE.md: never resume NaN); ones present at adoption/start are in nonfinite_known.
        new = [f for f in lb_nonfinite_files(run) if f not in st.get("nonfinite_known", [])]
        if new and not nonfinite_marker(rd):
            write_json(os.path.join(rd, NONFINITE_MARKER),
                       {"job": st["primary"]["job"], "where": "trainer",
                        "detail": f"{', '.join(new)} in {run['out_dir']}", "t": iso()})
            event(rd, "nonfinite", job=st["primary"]["job"], where="trainer", files=new)
        st["nonfinite_known"] = st.get("nonfinite_known", []) + new
    mk = nonfinite_marker(rd)
    if mk:
        tracked = {c["job"] for c in ([st["primary"]] if st.get("primary") else []) + st["race"]
                   + st["probes"]}
        if str(mk.get("job")) in tracked:
            halt_nonfinite(run, rd, st, mk)
            write_json(sp, st)
            return

    p = st.get("primary")
    if p is None:
        # tops up any allowed shape with no queued job (also picks up configs added to run.json)
        changed |= race_submit(run, rd, st, "no job" if not st["race"] else "new shapes")
        if changed:
            write_json(sp, st)
        return
    info = infos.get(p["job"]) or job_info(p["job"])
    state = info["state"]
    if info["restarts"] > p.get("restarts", 0):
        p["restarts"] = info["restarts"]
        p["pending_since"] = now()
        p["started"] = None
        event(rd, "preempted_requeued", job=p["job"], restarts=info["restarts"])
        notify(run, rd, f"job {p['job']} PREEMPTED (requeued, restart {info['restarts']}) on "
                        f"{cfg_str(cfgs[p['config']])}; will resume from checkpoint")
        changed = True

    if state in RUNNINGISH:
        # Only RUNNING is a start: a preempted job passes through COMPLETING on its way back to PENDING.
        if not p.get("started") and state == "RUNNING":
            p["started"] = now()
            p["pending_since"] = None
            event(rd, "running", job=p["job"], config=cfg_str(cfgs[p["config"]]))
            changed = True
        log = job_log(rd, p)
        limit = pol["stall_s"]
        if kind(run) == "liveband":
            # Progress, not log churn: a new train record in metrics.jsonl. Before this job's first
            # record (gates, ckpt load, torch.compile) the longer startup_stall_s applies.
            _, mt = lb_progress(run)
            t0 = p.get("started") or now()
            if mt > t0:
                quiet = now() - mt
            else:
                quiet, limit = now() - t0, pol["startup_stall_s"]
        else:
            # A missing log cannot be judged (job.sbatch writes it from the first line): only a log
            # that exists and has gone quiet counts.
            quiet = (now() - max(os.path.getmtime(log), p.get("started") or now())
                     if os.path.exists(log) else 0)
        # max_stalls counts stalls IN A ROW: a new checkpoint is proof the run got past them, so the
        # count restarts. (Before 2026-11-02 it was lifetime-cumulative, and arc_repro5000 was halted
        # by its 11th multi-node startup hang spread over 3 days of otherwise healthy training.)
        if st.get("stalls", 0) > 0:
            ck_step = progress_step(run, rd)
            if ck_step > st.get("stall_ckpt_step", -1):
                if "stall_ckpt_step" in st:
                    event(rd, "stalls_reset", stalls=st["stalls"], ckpt_step=ck_step)
                    st["stalls"] = 0
                st["stall_ckpt_step"] = ck_step
                changed = True
        if st.get("noprogress_failures") and progress_step(run, rd) > st.get("noprogress_ckpt_step", -1):
            if "noprogress_ckpt_step" in st:
                st["noprogress_failures"] = 0
            st["noprogress_ckpt_step"] = progress_step(run, rd)
            changed = True
        if state == "RUNNING" and p.get("started") and quiet > limit:
            sh(["scancel", p["job"]], check=False)
            st["stalls"] = st.get("stalls", 0) + 1
            event(rd, "stalled_cancelled", job=p["job"], quiet_s=int(quiet), stalls=st["stalls"],
                  config=cfg_str(cfgs[p["config"]]))
            notify(run, rd, f"job {p['job']} on {cfg_str(cfgs[p['config']])} silent {int(quiet) // 60} min: "
                            f"cancelled, racing again (stall {st['stalls']}/{pol['max_stalls']})",
                   urgent=kind(run) == "liveband")
            st["primary"] = None
            if st["stalls"] > pol["max_stalls"]:
                st["status"] = "halted"
                event(rd, "halted", job=p["job"], state="STALLED", stalls=st["stalls"], log=log)
                drop_queued_on_stop(rd, st)
            else:
                race_submit(run, rd, st, "primary stalled")
            write_json(sp, st)
            return
        # Move up only for more GPUs and only when the rest of the run finishes sooner by more than
        # switch_cost_s + switch_min_gain_s. The probe is a queued job: if it never starts it costs
        # nothing, and the old job keeps training until the probe holds its allocation.
        if p.get("started") and now() - max(p["started"], st.get("last_change", 0)) > pol["fit_dwell_s"] \
                and not st["probes"] and now() > st.get("next_probe_after", 0):
            cur = cfgs[p["config"]]
            ups = [j for j in range(len(cfgs)) if world(cfgs[j]) > world(cur)
                   and switch_gain_s(run, rd, p, cur, cfgs[j]) > pol["switch_min_gain_s"]]
            ups = [j for j in ups if not (st.get("submit_failed", {}).get(cfg_str(cfgs[j]), {})
                                          .get("retry_after", 0) > now())]
            for j in ups:
                q = try_submit(run, rd, st, j, "probe", primary_job=p["job"])
                if q:
                    st["probes"].append(q)
            if ups:
                event(rd, "probing_up", to=[cfg_str(cfgs[j]) for j in ups],
                      est_gain_s=int(max(switch_gain_s(run, rd, p, cur, cfgs[j]) for j in ups)))
                changed = True

    elif state in PENDINGISH:
        # Queued (first submit or a preemption requeue, which keeps its queue age): race every
        # other allowed shape alongside it. Nothing queued is ever cancelled here.
        if not p.get("pending_since"):
            p["pending_since"] = now()
            changed = True
        changed |= race_submit(run, rd, st, "primary queued")

    else:  # terminal
        step = progress_step(run, rd)
        total = max_steps_of(run)
        lb = kind(run) == "liveband"
        if state == "COMPLETED" and lost_race(rd, p["job"]):
            event(rd, "primary_lost_race", job=p["job"])
            st["primary"] = None
            race_submit(run, rd, st, "primary lost the start lock")
        elif state == "COMPLETED" and lb and total is not None and step < total \
                and step > st.get("early_exit_step", -1):
            # The LiveBand trainer saves and exits 0 on USR1/TERM: at the 7-day (or an adopted
            # job's 24 h) time limit, and on a preemption Slurm did not requeue. Not finished.
            st["early_exit_step"] = step
            event(rd, "resubmit_after", job=p["job"], state="COMPLETED_EARLY", step=step, steps=total)
            notify(run, rd, f"job {p['job']} exited 0 at step {step} < {total}: resubmitting", urgent=True)
            st["primary"] = None
            race_submit(run, rd, st, "primary exited before --steps")
        elif state == "COMPLETED" and lb and total is not None and step < total:
            # exited 0 again without any progress since the last early exit: a crash in disguise
            st["failures"] = st.get("failures", 0) + 1
            log = job_log(rd, p)
            if st["failures"] > pol["max_failures"]:
                st["status"] = "halted"
                event(rd, "halted", job=p["job"], state="COMPLETED_NO_PROGRESS", step=step, log=log)
                notify(run, rd, f"HALTED: job {p['job']} exited 0 at step {step} with no progress, "
                                f"see {log}", urgent=True)
            else:
                event(rd, "failed_resubmit", job=p["job"], state="COMPLETED_NO_PROGRESS", step=step,
                      failures=st["failures"], log=log)
                notify(run, rd, f"job {p['job']} exited 0 with no progress at step {step}, resubmitting "
                                f"({st['failures']}/{pol['max_failures']})", urgent=True)
                st["primary"] = None
                race_submit(run, rd, st, "primary exited 0 without progress")
        elif state == "COMPLETED":
            st["status"] = "done"
            event(rd, "finished", job=p["job"], ckpt_step=step)
            notify(run, rd, f"FINISHED at step {step}", urgent=lb)
        elif state == "CANCELLED":
            st["status"] = "paused"
            event(rd, "cancelled_externally", job=p["job"])
            notify(run, rd, f"job {p['job']} was cancelled outside the controller; run paused")
        elif state in ("TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"):
            event(rd, "resubmit_after", job=p["job"], state=state, ckpt_step=step)
            if lb:
                notify(run, rd, f"job {p['job']} {state} at step {step}: resubmitting", urgent=True)
            st["primary"] = None
            race_submit(run, rd, st, f"primary {state}")
        elif pol["shape_backoff"] and not lb and state != "UNKNOWN" and not job_made_progress(run, rd, p["job"]):
            if shape_failed(run, rd, st, p["config"], p["job"], state):
                st["status"] = "halted"
                event(rd, "halted", job=p["job"], state="NOPROGRESS_FAILURES", n=st["noprogress_failures"],
                      log=job_log(rd, p))
                notify(run, rd, f"HALTED: {st['noprogress_failures']} failures in a row without progress "
                                f"(last {state} on {cfg_str(cfgs[p['config']])}), see {job_log(rd, p)}", urgent=True)
            else:
                st["primary"] = None
                race_submit(run, rd, st, f"primary {state} without progress (shape charged)")
        else:  # FAILED, OUT_OF_MEMORY, UNKNOWN, ...
            st["failures"] = st.get("failures", 0) + 1
            log = job_log(rd, p)
            if st["failures"] > pol["max_failures"]:
                st["status"] = "halted"
                event(rd, "halted", job=p["job"], state=state, failures=st["failures"], log=log)
                notify(run, rd, f"HALTED after {st['failures']} failures (last {state}), see {log}",
                       urgent=True)
            else:
                event(rd, "failed_resubmit", job=p["job"], state=state, exit=info["exit"],
                      failures=st["failures"], log=log)
                notify(run, rd, f"job {p['job']} {state}, resubmitting "
                                f"({st['failures']}/{pol['max_failures']})", urgent=lb)
                st["primary"] = None
                race_submit(run, rd, st, f"primary {state}")
        changed = True
        changed |= drop_queued_on_stop(rd, st)

    if changed:
        write_json(sp, st)

    if pol.get("prune_ckpts", True):
        gone = prune_ckpts(rd, keep=pol.get("ckpt_keep", 2), min_age_s=pol.get("ckpt_prune_min_age_s", 600))
        if gone:
            event(rd, "ckpts_pruned", n=len(gone), freed_gb=round(sum(b for _, b in gone) / 1e9, 1),
                  files=[os.path.basename(p) for p, _ in gone])


HB_MAIN = ".controller_heartbeat"            # written by the login1 controller each pass
HB_WATCHDOG = ".controller_heartbeat_watchdog"   # written by watchdog takeover passes
WATCHDOG_NAME = "el:watchdog"
WATCHDOG_OFF = ".watchdog_off"                  # touch in ROOT to stop the watchdog (and its upkeep)
_tick_err_last = {}


def hb_age(name):
    try:
        return now() - os.path.getmtime(os.path.join(ROOT, name))
    except OSError:
        return None


def watchdog_flags():
    """Site overrides for watchdog.sbatch (its #SBATCH lines are Blue Vela's; command-line flags win)."""
    if SITE_NAME == "bluevela":
        return []
    f = SITE["cpu_job"] + [f"--export=ALL,ELASTIC_SITE={SITE_NAME},ELASTIC_HOME={HOME},ELASTIC_PY={SITE['python']}"]
    if SITE.get("watchdog_log"):
        f += ["-o", SITE["watchdog_log"]]
    return f


def ensure_watchdog():
    """Keep one el:watchdog job queued or running (2026-11-13: the controller died with login1's
    /proj at 06:48 and nothing noticed for 8 h). The watchdog keeps a singleton successor itself;
    this is the other half of the mutual watch."""
    if os.path.exists(os.path.join(ROOT, WATCHDOG_OFF)):
        return
    r = sh(["squeue", "--me", "-h", "-n", WATCHDOG_NAME, "-o", "%i"], check=False)
    if r.returncode == 0 and not (r.stdout or "").strip():
        sub = sh(["sbatch", "--parsable"] + watchdog_flags() + [os.path.join(HOME, "watchdog.sbatch")], check=False)
        print(f"[controller] watchdog submitted: {sub.stdout.strip() or sub.stderr.strip()[:200]}",
              flush=True)


def cmd_controller(a):
    os.makedirs(ROOT, exist_ok=True)
    watchdog = a.role == "watchdog"
    # The flock only excludes controllers on the SAME node: GPFS does not enforce it across nodes
    # (tested 2026-11-13: a cpu node acquired it while login1 held it). Login1 vs watchdog
    # exclusion is the heartbeat handover below.
    lock = open(os.path.join(ROOT, ".controller.lock" + (".watchdog" if watchdog else "")), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another controller holds the lock", file=sys.stderr)
        raise SystemExit(75)
    print(f"[controller] up, role={a.role}, root={ROOT}, interval={a.interval}s", flush=True)
    while True:
        if watchdog:
            age = hb_age(HB_MAIN)
            if age is not None and age < a.stale:
                print(f"[watchdog] main controller alive ({int(age)}s): not ticking", flush=True)
                return
        with open(os.path.join(ROOT, HB_WATCHDOG if watchdog else HB_MAIN), "w") as f:
            f.write(iso())
        if not watchdog:
            wa = hb_age(HB_WATCHDOG)
            if wa is not None and wa < a.interval + 60:
                # The watchdog was ticking for us: it sees our fresh heartbeat on its next pass
                # (and before each run); wait that out so two controllers never tick together.
                print(f"[controller] watchdog ticked {int(wa)}s ago: yielding one pass", flush=True)
                time.sleep(a.interval + 60)
            try:
                ensure_watchdog()
            except Exception as e:
                print(f"[controller] ensure_watchdog: {type(e).__name__}: {e}", flush=True)
        for name in sorted(os.listdir(ROOT)):
            rd = os.path.join(ROOT, name)
            if not os.path.isfile(os.path.join(rd, "run.json")):
                continue
            if watchdog:
                age = hb_age(HB_MAIN)
                if age is not None and age < a.stale:
                    print("[watchdog] main controller came back: stopping mid-pass", flush=True)
                    return
            try:
                tick(rd)
            except Exception as e:  # one broken run must not stop the others
                print(f"[controller] {name}: {type(e).__name__}: {e}", flush=True)
                # 2026-11-13: an sbatch that fails inside tick (a resubmit) used to be only a log
                # line. Alert, at most hourly per run.
                if not watchdog and now() - _tick_err_last.get(name, 0) > 3600:
                    _tick_err_last[name] = now()
                    try:
                        notify(load_run(rd), rd, f"CONTROLLER ERROR (resubmit may have failed): "
                                                 f"{type(e).__name__}: {str(e)[:300]}", urgent=True)
                    except Exception as e2:
                        print(f"[controller] {name}: notify failed: {e2}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


def alert(msg):
    """ntfy-only alert that needs no run dir (the watchdog; /proj may be unreadable)."""
    res = _send_ntfy(f"[{SITE['tag']}] {msg}"[:200], f"{msg}\n\ntime: {iso()}\nhost: {os.uname()[1]}\n")
    try:
        d = os.path.join(ROOT, "_watchdog")
        os.makedirs(d, exist_ok=True)
        event(d, "alert", msg=msg, ntfy=res)
    except OSError:
        print(f"[watchdog] alert (ntfy {res}): {msg}", flush=True)
    return res


def cmd_alert(a):
    alert(a.msg)


def cmd_watchdog(a):
    """Inside the el:watchdog cpu job (watchdog.sbatch), off login1. Every --interval s: if the login1
    controller's heartbeat is older than --stale s, alert (hourly) and run one controller pass
    ourselves (`controller --once --role watchdog`, a subprocess so a crash cannot kill us).
    Keeps a --dependency=singleton successor queued so a dead watchdog job is replaced."""
    global _NTFY_TOPIC
    try:
        _NTFY_TOPIC = open(NTFY_TOPIC_FILE).read().strip()
    except OSError:
        pass
    me = os.environ.get("SLURM_JOB_ID", "?")
    deadline = now() + a.hours * 3600
    last_alert = 0.0
    successor = False
    while now() < deadline:
        if os.path.exists(os.path.join(ROOT, WATCHDOG_OFF)):
            print("[watchdog] .watchdog_off present: exiting", flush=True)
            return
        if not successor:
            r = sh(["squeue", "--me", "-h", "-n", WATCHDOG_NAME, "-t", "PD", "-o", "%i"], check=False)
            if r.returncode == 0 and not (r.stdout or "").strip():
                sub = sh(["sbatch", "--parsable", "--dependency=singleton"] + watchdog_flags()
                         + [os.path.join(HOME, "watchdog.sbatch")], check=False)
                print(f"[watchdog] successor: {sub.stdout.strip() or sub.stderr.strip()[:200]}", flush=True)
            successor = r.returncode == 0
        age = hb_age(HB_MAIN)
        if age is None or age > a.stale:
            what = ("cannot read the controller heartbeat (/proj down?)" if age is None else
                    f"login1 controller heartbeat {int(age // 60)} min old")
            r = subprocess.run([sys.executable, "-I", os.path.join(HOME, "elastic.py"), "controller",
                                "--once", "--role", "watchdog", "--stale", str(a.stale)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               timeout=1800) if age is not None else None
            took = r is not None and r.returncode == 0
            if r is not None:
                print(r.stdout[-4000:], flush=True)
            if now() - last_alert > 3600:
                last_alert = now()
                alert(f"CONTROLLER DOWN: {what}. Watchdog job {me} "
                      f"{'is ticking every run in its place' if took else 'could NOT tick (rc ' + str(r and r.returncode) + ')'}. "
                      f"Restart on the login node: tmux new-session -d -s elastic '{SITE_ENV_PREFIX}bash {HOME}/controller.sh'")
        time.sleep(a.interval)


# ----------------------------------------------------------------------------- pool: idempotent shard work (2026-11-15)
# kind "pool" (CSAIL Option C encode): a run is a fixed list of SHARDS, each an idempotent command that
# writes atomically and ends with a done-file. The controller keeps WORKER jobs (1 GPU each) queued
# across every slot of the run (preemptible shared partitions included); a worker that starts claims
# shards one at a time by mkdir <run>/locks/<id> (atomic on NFS/GPFS), runs the command with the
# shard's vars substituted, touches the lock's heartbeat while it runs, and moves on until nothing is
# left or its time is nearly up. A preempted / killed / hung worker leaves a lock whose owner is not
# RUNNING (or whose heartbeat is older than stall_s): the controller deletes it and the shard is
# claimed again. A shard that fails max_shard_failures times is parked as BAD (alert), never looped.
#
# run.json:
#   {"name", "kind": "pool", "repo", "python",
#    "cmd": ["$PYTHON", "-I", "scripts/x.py", "--split", "{split}", "--shard", "{shard}"],
#    "done": "/abs/out/{split}/shard_{shard:04d}/DONE",      (str.format with the shard's vars)
#    "shards": [{"id": "train_0000", "vars": {"split": "train", "shard": 0}}, ...],
#    "slots": ["vs-h100", "vs-a100", ...]  (site slots),  "max_workers": 40, "max_pending_per_slot": 4,
#    "slot_caps": {"hai-l40s": 8}, "env": {...},
#    "slurm": {"cpus_per_gpu": 12, "mem_per_gpu_gb": 64, "time": "23:00:00"},
#    "policy": {"stall_s": 1200, "max_shard_failures": 3, "claim_cutoff_s": 1800}}

POOL_POLICY = {"stall_s": 1200, "max_shard_failures": 3, "claim_cutoff_s": 1800, "max_pending_per_slot": 4,
               "pending_cancel_grace_s": 300}


def pool_policy(run):
    return {**POOL_POLICY, **run.get("policy", {})}


def pool_done_path(run, sh_):
    return run["done"].format(**sh_["vars"])


def pool_shard_state(run, rd):
    """{id: 'done' | 'bad' | 'todo'}"""
    pol = pool_policy(run)
    out = {}
    for sh_ in run["shards"]:
        if os.path.exists(pool_done_path(run, sh_)):
            out[sh_["id"]] = "done"
        elif pool_fail_count(rd, sh_["id"]) >= pol["max_shard_failures"]:
            out[sh_["id"]] = "bad"
        else:
            out[sh_["id"]] = "todo"
    return out


def pool_bad_hosts(rd):
    """Hosts a worker declared bad (3 shards failed in a row there). Excluded from new submissions, and
    failures recorded on them do not count against a shard (2026-11-15: one segfaulting node, improbablex002,
    parked 6 healthy shards as BAD)."""
    try:
        with open(os.path.join(rd, "bad_hosts")) as f:
            return {l.strip() for l in f if l.strip()}
    except OSError:
        return set()


def pool_fail_count(rd, sid):
    bad_hosts = pool_bad_hosts(rd)
    try:
        with open(os.path.join(rd, "fail", sid)) as f:
            n = 0
            for l in f:
                m = re.search(r"\bhost (\S+)", l)
                if not (m and m.group(1) in bad_hosts):
                    n += 1
            return n
    except OSError:
        return 0


def pool_locks(rd):
    """{shard id: (owner job or None, heartbeat age s)}"""
    d = os.path.join(rd, "locks")
    out = {}
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        p = os.path.join(d, n)
        try:
            owner = open(os.path.join(p, "owner")).read().split()[0]
        except (OSError, IndexError):
            owner = None
        try:
            age = now() - max(os.path.getmtime(os.path.join(p, "owner")), os.path.getmtime(p))
        except OSError:
            try:
                age = now() - os.path.getmtime(p)
            except OSError:
                continue
        out[n] = (owner, age)
    return out


def pool_release(rd, sid):
    import shutil
    shutil.rmtree(os.path.join(rd, "locks", sid), ignore_errors=True)


def pool_my_jobs(run):
    """{job id: (state, slot, node)} for this run's live workers, from one squeue."""
    r = sh(["squeue", "--me", "-h", "-o", "%i|%T|%j|%N|%r"], check=False)
    out = {}
    if r.returncode != 0:
        raise RuntimeError(f"squeue failed: {r.stderr.strip()[:200]}")
    pre = f"el:{run['name']}:"
    for line in (r.stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[2].startswith(pre):
            out[parts[0]] = (parts[1], parts[2][len(pre):], parts[3] if len(parts) > 3 else "")
            reason = parts[4] if len(parts) > 4 else ""
            if parts[1] in PENDINGISH and ("held" in reason.lower() or "launch_failed" in reason.lower()):
                out[parts[0]] = ("HELD", out[parts[0]][1], reason)
    return out


def pool_sbatch_flags(run, rd, slot_name):
    s = run["slurm"]
    slots = {**SITE.get("slots", {}), **run.get("slots_def", {})}
    sl = slots[slot_name]
    f = ["-A", sl["account"], "-p", sl["partition"], f"--qos={sl['qos']}", "-N", "1"] + \
        ([f"--gres=gpu:{sl['gres']}:1"] if sl.get("gres") else []) + ["-c", str(sl.get("cpus", s["cpus_per_gpu"])),
         f"--mem={sl.get('mem_gb', s['mem_per_gpu_gb'])}G", "-t", sl.get("time", s["time"]),
         "--requeue", "--open-mode=append", "--signal=B:USR1@120",
         "-J", f"el:{run['name']}:{slot_name}", "-o", os.path.join(rd, "logs", "%j.out"),
         f"--export=ALL,ELASTIC_RUN={rd},ELASTIC_HOME={HOME},ELASTIC_SITE={SITE_NAME},ELASTIC_PY={SITE['python']}"]
    if sl.get("constraint"):
        f.append(f"--constraint={sl['constraint']}")
    excl = [x for x in (s.get("exclude") or "").split(",") if x]
    excl += sorted(h.split(".")[0] for h in pool_bad_hosts(rd) if h.split(".")[0] not in excl)
    if excl:
        f.append(f"--exclude={','.join(excl)}")
    return f


def pool_submit(run, rd, slot_name):
    r = sh(["sbatch", "--parsable"] + pool_sbatch_flags(run, rd, slot_name) + [os.path.join(HOME, "job_pool.sbatch")])
    jid = r.stdout.strip().split(";")[0]
    event(rd, "submitted", job=jid, slot=slot_name)
    return jid


def pool_free_gpus(run):
    """{slot: free GPUs of its type in its partition now} (a hint for where to queue first)."""
    slots = {**SITE.get("slots", {}), **run.get("slots_def", {})}
    r = sh(["scontrol", "show", "node", "-o"], check=False)
    free = {k: 0 for k in run["slots"]}
    for line in (r.stdout or "").splitlines():
        kv = dict(re.findall(r"(\w+)=(\S*)", line))
        state = kv.get("State", "").upper()
        if any(x in state for x in ("DOWN", "DRAIN", "FAIL", "MAINT", "RESERVED", "NOT_RESPONDING", "POWER")):
            continue
        parts = set(kv.get("Partitions", "").split(","))
        for k in run["slots"]:
            sl = slots[k]
            if sl["partition"] not in parts or not sl.get("gres"):
                continue
            t = re.escape(sl["gres"])
            cfg_ = re.search(rf"gres/gpu:{t}=(\d+)", kv.get("CfgTRES", ""))
            al = re.search(rf"gres/gpu:{t}=(\d+)", kv.get("AllocTRES", ""))
            if cfg_:
                free[k] += int(cfg_.group(1)) - (int(al.group(1)) if al else 0)
    return free


def tick_pool(run, rd, st):
    pol = pool_policy(run)
    states = pool_shard_state(run, rd)
    todo = [k for k, v in states.items() if v == "todo"]
    bad = sorted(k for k, v in states.items() if v == "bad")
    jobs = pool_my_jobs(run)
    held = [j for j, v in jobs.items() if v[0] == "HELD"]
    for j in held:                       # a held requeue never starts on its own (e.g. launch failure)
        sh(["scancel", j], check=False)
        event(rd, "held_cancelled", job=j, slot=jobs[j][1], reason=jobs[j][2])
        jobs.pop(j)
    running = {j: v for j, v in jobs.items() if v[0] in RUNNINGISH}
    pending = {j: v for j, v in jobs.items() if v[0] in PENDINGISH}
    # stale / hung locks
    for sid, (owner, age) in pool_locks(rd).items():
        if states.get(sid) == "done":
            pool_release(rd, sid)
            continue
        if owner is None:
            if age > 120:
                pool_release(rd, sid); event(rd, "lock_released", shard=sid, why="no owner")
        elif owner not in running:
            pool_release(rd, sid); event(rd, "lock_released", shard=sid, why=f"owner {owner} not running")
        elif age > pol["stall_s"]:
            sh(["scancel", owner], check=False)
            pool_release(rd, sid)
            event(rd, "stall", shard=sid, job=owner, node=running[owner][2], heartbeat_age_s=int(age))
    newly_bad = [b for b in bad if b not in st.get("bad_known", [])]
    if newly_bad:
        st["bad_known"] = bad
        notify(run, rd, f"{len(newly_bad)} shard(s) failed {pol['max_shard_failures']}x and are parked BAD: "
                        f"{newly_bad[:10]}", urgent=True)
    if not todo:
        for j in pending:
            sh(["scancel", j], check=False)
        st["status"] = "done" if not bad else "done_with_bad"
        event(rd, "pool_done", shards=len(states), bad=bad)
        notify(run, rd, f"POOL DONE: {len(states) - len(bad)}/{len(states)} shards"
                        + (f", {len(bad)} BAD" if bad else ""), urgent=True)
        return True
    # enough workers already running for what is left: drop queued ones
    if len(running) >= len(todo):
        for j in pending:
            sh(["scancel", j], check=False)
        if pending:
            event(rd, "cancel_pending", jobs=sorted(pending), why="running workers cover the remaining shards")
        return False
    target = min(int(run.get("max_workers", 32)), len(todo))
    want = target - len(running) - len(pending)
    if want <= 0:
        return False
    caps = run.get("slot_caps", {})
    per_slot = {k: 0 for k in run["slots"]}
    per_slot_pend = {k: 0 for k in run["slots"]}
    for _, (state, slot, _) in jobs.items():
        if slot in per_slot:
            per_slot[slot] += 1
            per_slot_pend[slot] += state in PENDINGISH
    try:
        free = pool_free_gpus(run)
    except Exception:
        free = {k: 0 for k in run["slots"]}
    order = sorted(run["slots"], key=lambda k: (-free.get(k, 0), run["slots"].index(k)))
    sub = []

    def try_submit(k):
        try:
            jid = pool_submit(run, rd, k)
        except RuntimeError as e:
            event(rd, "submit_failed", slot=k, err=str(e)[:300])
            per_slot_pend[k] = 10 ** 6          # this slot refuses us: skip it for the rest of the tick
            return False
        sub.append(f"{jid}:{k}")
        per_slot[k] += 1
        per_slot_pend[k] += 1
        return True

    def cap_room(k):
        return caps[k] - per_slot[k] if k in caps else 10 ** 6

    # 1. fill the GPUs that are free right now, most free first
    for k in order:
        n = min(want, free.get(k, 0) - per_slot_pend[k], cap_room(k))
        while n > 0 and try_submit(k):
            n -= 1
            want -= 1
    # 2. the rest waits in every slot's queue (up to max_pending_per_slot each), round robin
    while want > 0:
        progressed = False
        for k in order:
            if want <= 0:
                break
            if cap_room(k) <= 0 or per_slot_pend[k] >= pol["max_pending_per_slot"]:
                continue
            if try_submit(k):
                want -= 1
                progressed = True
        if not progressed:
            break
    if sub:
        event(rd, "workers", submitted=sub, running=len(running), pending=len(pending), todo=len(todo))
    return bool(sub)


def pool_status(run, rd):
    states = pool_shard_state(run, rd)
    c = {k: sum(1 for v in states.values() if v == k) for k in ("done", "todo", "bad")}
    try:
        jobs = pool_my_jobs(run)
    except RuntimeError:
        jobs = {}
    by = {}
    for _, (state, slot, _) in jobs.items():
        by.setdefault(slot, {}).setdefault(state, 0)
        by[slot][state] += 1
    locks = pool_locks(rd)
    print(f"  shards: {c['done']}/{len(states)} done, {c['todo']} todo ({len(locks)} claimed), {c['bad']} BAD")
    print(f"  workers: " + (", ".join(f"{k} {v}" for k, v in sorted(by.items())) or "none"))


def cmd_pool_worker(a):
    """Inside job_pool.sbatch: claim shards (mkdir lock), run each, until none is left or time is short."""
    import random
    import signal
    import threading
    rd = a.run
    run = load_run(rd)
    pol = pool_policy(run)
    me = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    host = os.uname()[1]
    os.makedirs(os.path.join(rd, "locks"), exist_ok=True)
    os.makedirs(os.path.join(rd, "fail"), exist_ok=True)
    # a requeued job keeps its id: locks from our previous incarnation are ours to drop
    for sid, (owner, _) in pool_locks(rd).items():
        if owner == me:
            pool_release(rd, sid)
    left = sh(["squeue", "-h", "-j", me, "-o", "%L"], check=False).stdout.strip()
    secs = None
    m = re.match(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", left or "")
    if m:
        d_, h_, m_, s_ = (int(x) if x else 0 for x in m.groups())
        secs = ((d_ * 24 + h_) * 60 + m_) * 60 + s_
    deadline = now() + (secs if secs else 10 ** 9)
    child = {"p": None, "sid": None}
    stop = {"v": False}

    def on_sig(signum, frame):
        stop["v"] = True
        p = child["p"]
        print(f"[pool] {me}: signal {signum}; stopping shard {child['sid']}", flush=True)
        if p and p.poll() is None:
            p.terminate()

    for sg in (signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(sg, on_sig)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in run.get("env", {}).items()})
    shards = list(run["shards"])
    random.Random(me).shuffle(shards)
    consecutive_fail, n_done = 0, 0
    failed_here = set()       # never retry a shard on the worker it just failed on: let another node try it
    event(rd, "worker_start", job=me, host=host, time_left_s=secs)
    while not stop["v"]:
        if now() > deadline - pol["claim_cutoff_s"]:
            event(rd, "worker_exit", job=me, why="time", done=n_done)
            break
        claimed = None
        for sh_ in shards:
            if sh_["id"] in failed_here or os.path.exists(pool_done_path(run, sh_)) \
                    or pool_fail_count(rd, sh_["id"]) >= pol["max_shard_failures"]:
                continue
            lk = os.path.join(rd, "locks", sh_["id"])
            try:
                os.mkdir(lk)
            except OSError:
                continue
            with open(os.path.join(lk, "owner"), "w") as f:
                f.write(f"{me} {host} {iso()}\n")
            if os.path.exists(pool_done_path(run, sh_)):     # finished between the check and the mkdir
                pool_release(rd, sh_["id"]); continue
            claimed = sh_
            break
        if claimed is None:
            event(rd, "worker_exit", job=me, why="no shards left", done=n_done)
            break
        sid = claimed["id"]
        cmd = [c.replace("$PYTHON", run["python"]).format(**claimed["vars"]) for c in run["cmd"]]
        print(f"[pool] {me} {host}: shard {sid}: {' '.join(cmd)}", flush=True)
        t0 = now()
        child["sid"] = sid
        child["p"] = subprocess.Popen(cmd, cwd=run["repo"], env=env)
        lk_owner = os.path.join(rd, "locks", sid, "owner")

        def beat():
            while child["p"] is not None and child["p"].poll() is None:
                try:
                    os.utime(lk_owner)
                except OSError:
                    pass
                time.sleep(30)
        threading.Thread(target=beat, daemon=True).start()
        rc = child["p"].wait()
        child["p"] = None
        ok = rc == 0 and os.path.exists(pool_done_path(run, claimed))
        if ok:
            n_done += 1; consecutive_fail = 0
            event(rd, "shard_done", shard=sid, job=me, host=host, secs=round(now() - t0, 1))
        elif stop["v"]:
            event(rd, "shard_interrupted", shard=sid, job=me, host=host)
        else:
            consecutive_fail += 1
            failed_here.add(sid)
            with open(os.path.join(rd, "fail", sid), "a") as f:
                f.write(f"{iso()} job {me} host {host} rc {rc}\n")
            event(rd, "shard_failed", shard=sid, job=me, host=host, rc=rc)
        pool_release(rd, sid)
        if consecutive_fail >= 3:
            if host not in pool_bad_hosts(rd):
                with open(os.path.join(rd, "bad_hosts"), "a") as f:
                    f.write(host + "\n")
            event(rd, "bad_node", job=me, host=host, why="3 shards failed in a row; host excluded")
            raise SystemExit(3)


# ----------------------------------------------------------------------------- CLI

def cmd_start(a):
    rd = run_dir(a.run)
    run = load_run(rd)
    errs = validate(run)
    if errs:
        raise SystemExit("run.json invalid:\n  " + "\n  ".join(errs))
    for sub in ("ckpt", "logs"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)
    st = read_json(os.path.join(rd, "state.json"))
    if st and st.get("status") == "active" and st.get("primary"):
        raise SystemExit(f"already active with job {st['primary']['job']}")
    if kind(run) == "pool":
        for sub in ("locks", "fail"):
            os.makedirs(os.path.join(rd, sub), exist_ok=True)
        if st and st.get("status") == "active" and not a.dry_run:
            raise SystemExit("pool already active")
        st = {"status": "active", "started": now(), "bad_known": []}
        if a.dry_run:
            print("DRY RUN: would submit, per slot:")
            for k in run["slots"]:
                print("  sbatch --parsable " + " ".join(shlex.quote(x) for x in pool_sbatch_flags(run, rd, k))
                      + " " + os.path.join(HOME, "job_pool.sbatch"))
            states = pool_shard_state(run, rd)
            print(f"shards: {len(states)} ({sum(v == 'done' for v in states.values())} already done)")
            return
        tick_pool(run, rd, st)
        write_json(os.path.join(rd, "state.json"), st)
        event(rd, "started", shards=len(run["shards"]), slots=run["slots"])
        return
    if kind(run) == "liveband":
        dup = lb_live_jobs(run)
        if dup:
            raise SystemExit(f"refused: live job(s) already train {run['out_dir']}: {dup}. "
                             f"Use `elastic.py adopt {run['name']} <jobid>` instead.")
    clear_nonfinite_marker(rd)
    st = {"status": "active", "primary": None, "probes": [], "race": [], "failures": 0,
          "last_change": now(), "probe_backoff_s": run["policy"]["probe_backoff_s"]}
    if kind(run) == "liveband":
        st["nonfinite_known"] = lb_nonfinite_files(run)
    if a.config is not None:
        st["primary"] = submit(run, rd, a.config, "primary")
    race_submit(run, rd, st, "start")
    write_json(os.path.join(rd, "state.json"), st)
    event(rd, "started", configs=[cfg_str(c) for c in run["configs"]],
          global_bs=run.get("global_bs"), micro_bs=run.get("micro_bs"))


def cmd_status(a):
    names = [a.run] if a.run else sorted(os.listdir(ROOT))
    hb = os.path.join(ROOT, ".controller_heartbeat")
    if os.path.exists(hb):
        print(f"controller heartbeat: {open(hb).read()} ({int(now() - os.path.getmtime(hb))}s ago)")
    else:
        print("controller heartbeat: NONE")
    for name in names:
        rd = run_dir(name)
        if not os.path.isfile(os.path.join(rd, "run.json")):
            continue
        run = load_run(rd)
        st = read_json(os.path.join(rd, "state.json"), {})
        if kind(run) == "pool":
            print(f"{run['name']:<28} {st.get('status', '-')} (pool)")
            pool_status(run, rd)
            continue
        if kind(run) == "liveband":
            step, mt = lb_progress(run)
            line = (f"{run['name']:<28} {st.get('status', '-'):<7} step={step}/{max_steps_of(run)} "
                    f"(logged {int(now() - mt) if mt else '-'}s ago)")
        else:
            _, step = pick_ckpt(os.path.join(rd, "ckpt"))
            line = f"{run['name']:<28} {st.get('status', '-'):<7} ckpt_step={step}"
        mk = nonfinite_marker(rd)
        if mk:
            line += f"  NONFINITE[job {mk.get('job')} {mk.get('where')}]"
        for role, j in [("primary", st.get("primary")), ("probe", st.get("probe"))] + \
                       [("probe", q) for q in st.get("probes", [])] + \
                       [("race", r) for r in st.get("race", [])]:
            if j:
                info = job_info(j["job"])
                line += (f"  {role}={j['job']}[{cfg_str(run['configs'][j['config']])} "
                         f"{info['state']} restarts={info['restarts']}]")
        print(line)


def cmd_pause(a):
    rd = run_dir(a.run)
    sp = os.path.join(rd, "state.json")
    st = read_json(sp)
    st["status"] = "paused"
    if kind(load_run(rd)) == "pool":
        run = load_run(rd)
        pend = [j for j, v in pool_my_jobs(run).items() if v[0] in PENDINGISH]
        for j in pend:
            sh(["scancel", j], check=False)
        write_json(sp, st)
        event(rd, "paused", cancelled=pend, note="running workers finish their current shard and keep claiming; scancel them to stop")
        return
    # A paused run must not start later on its own (2026-10-27: paused runs' requeued and probe
    # jobs were still in the queue). Anything not yet RUNNING is cancelled; a running primary is
    # only touched with --save-stop.
    cancelled = []
    for r in st.get("race", []):
        sh(["scancel", r["job"]], check=False)
        cancelled.append(r["job"])
    st["race"] = []
    for q in st.get("probes", []) + ([st["probe"]] if st.get("probe") else []):
        sh(["scancel", q["job"]], check=False)
        cancelled.append(q["job"])
    st["probe"], st["probes"] = None, []
    p = st.get("primary")
    if p and job_info(p["job"])["state"] not in RUNNINGISH:
        sh(["scancel", p["job"]], check=False)
        cancelled.append(p["job"])
        st["config_hint"] = p["config"]
        st["primary"] = None
    elif p and a.save_stop:
        # LiveBand: USR1 is "checkpoint and exit 0" (an adopted plain job has no USR2 trap)
        sig = "USR1" if kind(load_run(rd)) == "liveband" else "USR2"
        sh(["scancel", f"--signal={sig}", "--batch", p["job"]], check=False)
    write_json(sp, st)
    event(rd, "paused", save_stop=a.save_stop, cancelled=cancelled)


def cmd_resume(a):
    rd = run_dir(a.run)
    sp = os.path.join(rd, "state.json")
    st = read_json(sp)
    st["status"] = "active"
    p = st.get("primary")
    if p and job_info(p["job"])["state"] not in PENDINGISH | RUNNINGISH:
        st["config_hint"] = p["config"]
        st["primary"] = None          # controller resubmits on its next tick
    st["failures"] = 0
    st["stalls"] = 0              # a manual resume is a fresh start for the in-a-row stall count
    st.pop("stall_ckpt_step", None)
    clear_nonfinite_marker(rd)    # the user looked at the NaN halt; prepare re-checks every resume
    write_json(sp, st)
    event(rd, "resumed")


def cmd_simulate_preempt(a):
    """What Blue Vela does on preemption: USR1 to the batch shell, grace, then requeue."""
    rd = run_dir(a.run)
    st = read_json(os.path.join(rd, "state.json"))
    jid = st["primary"]["job"]
    _, before = pick_ckpt(os.path.join(rd, "ckpt"))
    sh(["scancel", "--signal=USR1", "--batch", jid])
    event(rd, "sim_preempt_signal", job=jid, ckpt_step_before=before)
    deadline = now() + a.grace
    while now() < deadline:
        _, step = pick_ckpt(os.path.join(rd, "ckpt"))
        if step > before:
            break
        time.sleep(5)
    _, step = pick_ckpt(os.path.join(rd, "ckpt"))
    sh(["scontrol", "requeue", jid])
    event(rd, "sim_preempt_requeue", job=jid, ckpt_step=step, saved=step > before)


def prepare_liveband(run, rd):
    """Inside a job_liveband.sbatch job. The trainer resumes from <out>/ckpt_last.pt by itself, so
    this only guards it: finite weights (scripts/check_ckpt_finite.py, as for every SA3 resume), and
    never a silent from-scratch start of a run that has progress. A non-finite ckpt_last is NOT
    renamed (the trainer would then start from step 0); the run halts with a NONFINITE marker."""
    out = run["out_dir"]
    jid = os.environ.get("SLURM_JOB_ID")
    os.makedirs(out, exist_ok=True)
    ck = os.path.join(out, "ckpt_last.pt")
    step, _ = lb_progress(run)
    if os.path.exists(ck):
        status, detail = (check_finite(run, ck) if run["policy"].get("check_ckpt_finite", True)
                          else ("unknown", "check disabled by policy"))
        if status == "finite":
            event(rd, "ckpt_finite", ckpt=ck, detail=detail)
        elif status == "nonfinite":
            msg = f"{ck} is non-finite: {detail}"
            write_json(os.path.join(rd, NONFINITE_MARKER),
                       {"job": jid, "where": "prepare", "detail": msg, "t": iso()})
            event(rd, "nonfinite", job=jid, where="prepare", detail=msg)
            raise SystemExit(f"FATAL: {msg}")
        elif status == "unreadable":
            event(rd, "prepare_failed", job=jid, detail=f"{ck} unreadable: {detail}")
            raise SystemExit(f"FATAL: {ck} unreadable: {detail}")
        else:
            event(rd, "ckpt_check_unavailable", ckpt=ck, detail=detail)
    else:
        milestones = [f for f in os.listdir(out) if re.fullmatch(r"step\d+\.pt", f)]
        if step > 0 or milestones:
            msg = (f"{ck} missing but the run has progress (logged step {step}, "
                   f"{len(milestones)} milestone(s)): refusing a from-scratch start")
            event(rd, "prepare_failed", job=jid, detail=msg)
            raise SystemExit(f"FATAL: {msg}")
    env = {**run.get("env", {}), "OUT": out, "LB_ROOT": run["repo"]}
    lines = [f"export {k}={shlex.quote(str(v))}" for k, v in env.items()]
    lines += [f"REPO={shlex.quote(run['repo'])}", f"SCRIPT={shlex.quote(run['script'])}",
              f"RESUME_STEP={step}",
              "PRE_LAUNCH=(" + " ".join(shlex.quote(c) for c in run.get("pre_launch", [])) + ")",
              "ARGS=(" + " ".join(shlex.quote(x) for x in run["args"]) + ")"]
    print("\n".join(lines))
    event(rd, "job_prepare", job=jid, config=f"1x{os.environ.get('SLURM_GPUS_ON_NODE', '?')}gpu",
          resume_ckpt=ck if os.path.exists(ck) else None, resume_step=step,
          restart=os.environ.get("SLURM_RESTART_COUNT", "0"))


def lb_live_jobs(run, exclude=()):
    """[(jobid, name, state)] of this user's non-terminal jobs that would train run['out_dir']:
    elastic jobs of this run (el:<name>:...) and plain train_data_ddp.sh jobs whose log header
    names this RUN (or, still pending, whose name is in run['plain_job_names'])."""
    r = sh(["squeue", "--me", "-h", "-o", "%i|%j|%T|%o"], check=False)
    out = []
    for line in (r.stdout or "").splitlines():
        parts = line.split("|")
        if len(parts) < 4 or parts[0] in exclude:
            continue
        jid, name, state, cmd = parts[:4]
        if name.startswith(f"el:{run['name']}:"):
            out.append((jid, name, state))
        elif cmd.endswith("train_data_ddp.sh") and name in run.get("plain_job_names", []):
            out.append((jid, name, state))
    return out


def cmd_adopt(a):
    """Put a live plain-sbatch LiveBand job under the controller without a second submit: it becomes
    the run's primary and holds the start lock, so no elastic job can train the same out_dir while it
    runs. When it ends (FAILED, a dead requeue, time limit, ...) the controller resubmits."""
    rd = run_dir(a.run)
    run = load_run(rd)
    if kind(run) != "liveband":
        raise SystemExit("adopt is for kind=liveband runs")
    errs = validate(run)
    if errs:
        raise SystemExit("run.json invalid:\n  " + "\n  ".join(errs))
    st = read_json(os.path.join(rd, "state.json"))
    if st and st.get("status") == "active" and st.get("primary"):
        raise SystemExit(f"already active with job {st['primary']['job']}")
    info = job_info(a.job)
    if info["state"] not in PENDINGISH | RUNNINGISH:
        raise SystemExit(f"job {a.job} is {info['state']}: nothing to adopt (use start)")
    r = sh(["scontrol", "show", "job", "-o", a.job], check=False)
    kv = dict(re.findall(r"(\w+)=(\S*)", r.stdout))
    if not kv.get("Command", "").endswith(os.path.basename(run["script"])):
        raise SystemExit(f"job {a.job} runs {kv.get('Command')}, not {run['script']}")
    if kv.get("JobName") not in run.get("plain_job_names", []):
        raise SystemExit(f"job {a.job} name {kv.get('JobName')} not in run.json plain_job_names")
    log = kv.get("StdOut", "")
    if info["state"] == "RUNNING":
        hdr = [l for l in open(log, errors="replace") if l.startswith("=== ")] if os.path.exists(log) else []
        if not hdr or f" {run['env']['RUN']} job {a.job} " not in hdr[-1]:
            raise SystemExit(f"log {log} does not show RUN={run['env']['RUN']} job {a.job}: "
                             f"{hdr[-1].strip() if hdr else 'no header'}")
    others = [j for j in lb_live_jobs(run) if j[0] != a.job]
    if others:
        raise SystemExit(f"refused: other live job(s) for this run: {others}")
    holder = lock_holder(rd)
    if holder and holder != a.job and job_info(holder)["state"] in RUNNINGISH:
        raise SystemExit(f"start lock held by running job {holder}")
    prim = {"job": a.job, "config": 0, "submitted": now(), "restarts": info["restarts"],
            "pending_since": None if info["state"] == "RUNNING" else now(),
            "started": now() if info["state"] == "RUNNING" else None,
            "log": log, "adopted": True}
    new = {"status": "active", "primary": prim, "probes": [], "race": [], "failures": 0,
           "last_change": now(), "probe_backoff_s": run["policy"]["probe_backoff_s"],
           "nonfinite_known": lb_nonfinite_files(run)}
    step, mt = lb_progress(run)
    print(f"adopt {run['name']}: job {a.job} ({kv.get('JobName')}, {info['state']}, restarts "
          f"{info['restarts']}), log {log}, logged step {step}/{max_steps_of(run)}, lock "
          f"{holder or 'free'} -> {a.job}")
    if a.dry_run:
        print("dry run: nothing written")
        return
    for sub in ("ckpt", "logs"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)
    lock = os.path.join(rd, ".train_lock")
    os.makedirs(lock, exist_ok=True)
    with open(os.path.join(lock, "job"), "w") as f:
        f.write(a.job + "\n")
    write_json(os.path.join(rd, "state.json"), new)
    event(rd, "adopted", job=a.job, name=kv.get("JobName"), state=info["state"], log=log, step=step)


def cmd_prepare(a):
    """Inside a job: resolve accum + resume checkpoint, print shell exports for job.sbatch."""
    rd = run_dir(a.run)
    run = load_run(rd)
    if kind(run) == "liveband":
        return prepare_liveband(run, rd)
    n = int(os.environ.get("SLURM_NNODES", "1"))
    g = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0)
    if g == 0:
        raise SystemExit("SLURM_GPUS_ON_NODE unset: no GPUs in this allocation")
    mb = batch_for(run, [n, g])
    micro, accum = mb if mb else (run["micro_bs"], None)
    if accum is None:
        raise SystemExit(f"global_bs {run['global_bs']} not divisible by micro {run['micro_bs']}"
                         f" x {n * g} GPUs")
    ckdir = os.path.join(rd, "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    ck, step, skipped = pick_finite_ckpt(run, rd, ckdir)
    if skipped:
        names = ", ".join(f"{b} ({s})" for b, s, _ in skipped)
        if ck is None:
            # Never fall through to a from-scratch start when the run HAS checkpoints, all bad.
            msg = f"no finite checkpoint left to resume from; skipped {names}"
            if any(s == "nonfinite" for _, s, _ in skipped):
                write_json(os.path.join(rd, NONFINITE_MARKER),
                           {"job": os.environ.get("SLURM_JOB_ID"), "where": "prepare",
                            "detail": msg, "t": iso()})
                event(rd, "nonfinite", job=os.environ.get("SLURM_JOB_ID"), where="prepare",
                      detail=msg)
            else:
                event(rd, "prepare_failed", job=os.environ.get("SLURM_JOB_ID"), detail=msg)
            raise SystemExit(f"FATAL: {msg}")
        notify(run, rd, f"resume skipped {names}; resuming from "
                        f"{os.path.basename(ck) if ck else 'scratch'}")
    args = list(run["args"]) + ["--batch_size", str(micro),
                                "--accum_batches", str(accum), "--save_dir", ckdir]
    if ck:
        args += ["--ckpt_path", ck]
    env = {**run.get("env", {}), "ELASTIC_GLOBAL_BS": str(run["global_bs"]),
           "ELASTIC_MICRO_BS": str(micro),
           "ELASTIC_SAVE_EVERY_S": str(int(run["policy"]["save_every_s"])),
           "ELASTIC_GRACE_SAVE_EVERY_S": str(int(run["policy"]["grace_save_every_s"]))}
    out = [f"export {k}={shlex.quote(str(v))}" for k, v in env.items()]
    out += [f"REPO={shlex.quote(run['repo'])}", f"PYBIN={shlex.quote(run['python'])}",
            f"SCRIPT={shlex.quote(run['script'])}", f"NNODES={n}", f"GPN={g}",
            f"ACCUM={accum}", f"RESUME_STEP={step}",
            f"GPU_PREFLIGHT={1 if SITE.get('gpu_count_preflight') else 0}",
            "PRE_LAUNCH=(" + " ".join(shlex.quote(c) for c in run.get("pre_launch", [])) + ")",
            "ARGS=(" + " ".join(shlex.quote(x) for x in args) + ")"]
    print("\n".join(out))
    event(rd, "job_prepare", job=os.environ.get("SLURM_JOB_ID"), config=cfg_str([n, g]),
          accum=accum, resume_ckpt=os.path.basename(ck) if ck else None, resume_step=step,
          restart=os.environ.get("SLURM_RESTART_COUNT", "0"))


def cmd_event(a):
    event(run_dir(a.run), a.kind, **dict(kv.split("=", 1) for kv in a.kv))


def cmd_notify(a):
    rd = run_dir(a.run)
    notify(load_run(rd), rd, a.msg)


def cmd_test_notify(a):
    """Exercise every channel without a run dir."""
    run = {"name": "test", "mail": a.to or mail_to({}), "slurm": dict(SLURM_DEFAULTS)}
    os.makedirs(os.path.join(ROOT, "_notify_test"), exist_ok=True)
    notify(run, os.path.join(ROOT, "_notify_test"), a.msg)


def cmd_fit(a):
    """Read-only: the shapes a run races and how many nodes could host each right now (a hint only;
    the race lets Slurm decide)."""
    names = [a.run] if a.run else sorted(n for n in os.listdir(ROOT)
                                         if os.path.isfile(os.path.join(ROOT, n, "run.json")))
    for name in names:
        rd = run_dir(name)
        run = load_run(rd)
        cap = capacity(run)
        print(f"{name}: races {len(run['configs'])} shape(s)")
        for cfg in run["configs"]:
            mb = batch_for(run, cfg)
            print(f"    {cfg_str(cfg)}: {hosts(run, cfg, cap)} node(s) look free for {cfg[1]} GPUs; "
                  f"micro {mb and mb[0]} x accum {mb and mb[1]}")


def cmd_exclude_nodes(a):
    """Add nodes to `slurm.exclude` of every run under ROOT and to every queued `el:` job of ours.
    Called by job.sbatch's multi-node preflight; safe to run by hand."""
    nodes = [n for n in a.nodes.split(",") if n]
    for name in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, name, "run.json")
        if not os.path.isfile(p):
            continue
        j = read_json(p)
        s = j.setdefault("slurm", {})
        ex = [x for x in (s.get("exclude") or "").split(",") if x]
        add = [n for n in nodes if n not in ex]
        if add:
            s["exclude"] = ",".join(ex + add)
            write_json(p, j)
            event(os.path.join(ROOT, name), "nodes_excluded", nodes=add)
    r = sh(["squeue", "--me", "-h", "-t", "PD", "-o", "%i %j"], check=False)
    for line in (r.stdout or "").splitlines():
        jid, _, jname = line.partition(" ")
        if not jname.startswith("el:"):
            continue
        cur = sh(["scontrol", "show", "job", jid], check=False).stdout or ""
        m = re.search(r"ExcNodeList=(\S*)", cur)
        have = [x for x in (m.group(1) if m else "").split(",") if x and x != "(null)"]
        want = have + [n for n in nodes if n not in have]
        if want != have:
            sh(["scontrol", "update", f"jobid={jid}", f"ExcNodeList={','.join(want)}"], check=False)


def cmd_prune(a):
    """Show (default) or do (--apply) the checkpoint pruning for one run."""
    rd = run_dir(a.run)
    pol = load_run(rd)["policy"]
    gone = prune_ckpts(rd, keep=pol.get("ckpt_keep", 2), min_age_s=pol.get("ckpt_prune_min_age_s", 600),
                       dry_run=not a.apply)
    for p, b in gone:
        print(f"{'deleted' if a.apply else 'would delete'} {os.path.basename(p)}  ({b / 1e9:.1f} GB)")
    print(f"{len(gone)} file(s), {sum(b for _, b in gone) / 1e9:.1f} GB" + ("" if a.apply else " (dry run)"))
    if a.apply and gone:
        event(rd, "ckpts_pruned", n=len(gone), freed_gb=round(sum(b for _, b in gone) / 1e9, 1),
              files=[os.path.basename(p) for p, _ in gone], manual=True)


def cmd_pick_ckpt(a):
    ck, step = pick_ckpt(a.dir)
    print(f"{step} {ck}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("start"); s.add_argument("run"); s.add_argument("--config", type=int, default=None,
                                                                   help="config index; default: best fit now")
    s.add_argument("--dry-run", action="store_true", help="pool: print the sbatch lines, submit nothing")
    s.set_defaults(fn=cmd_start)
    s = sp.add_parser("pool-worker"); s.add_argument("run"); s.set_defaults(fn=cmd_pool_worker)
    s = sp.add_parser("status"); s.add_argument("run", nargs="?"); s.set_defaults(fn=cmd_status)
    s = sp.add_parser("pause"); s.add_argument("run"); s.add_argument("--save-stop", action="store_true")
    s.set_defaults(fn=cmd_pause)
    s = sp.add_parser("resume"); s.add_argument("run"); s.set_defaults(fn=cmd_resume)
    s = sp.add_parser("controller"); s.add_argument("--interval", type=int, default=120)
    s.add_argument("--once", action="store_true")
    s.add_argument("--role", choices=["main", "watchdog"], default="main")
    s.add_argument("--stale", type=int, default=600, help="watchdog role: main heartbeat age that counts as dead")
    s.set_defaults(fn=cmd_controller)
    s = sp.add_parser("watchdog"); s.add_argument("--interval", type=int, default=120)
    s.add_argument("--stale", type=int, default=600); s.add_argument("--hours", type=float, default=23.5)
    s.set_defaults(fn=cmd_watchdog)
    s = sp.add_parser("alert"); s.add_argument("msg"); s.set_defaults(fn=cmd_alert)
    s = sp.add_parser("adopt"); s.add_argument("run"); s.add_argument("job")
    s.add_argument("--dry-run", action="store_true"); s.set_defaults(fn=cmd_adopt)
    s = sp.add_parser("simulate-preempt"); s.add_argument("run")
    s.add_argument("--grace", type=int, default=600); s.set_defaults(fn=cmd_simulate_preempt)
    s = sp.add_parser("prepare"); s.add_argument("run"); s.set_defaults(fn=cmd_prepare)
    s = sp.add_parser("event"); s.add_argument("run"); s.add_argument("kind")
    s.add_argument("kv", nargs="*"); s.set_defaults(fn=cmd_event)
    s = sp.add_parser("notify"); s.add_argument("run"); s.add_argument("msg"); s.set_defaults(fn=cmd_notify)
    s = sp.add_parser("test-notify"); s.add_argument("--to", default=None, help="default: site 'mail' or $ELASTIC_MAIL")
    s.add_argument("--msg", default="notification test"); s.set_defaults(fn=cmd_test_notify)
    s = sp.add_parser("pick-ckpt"); s.add_argument("dir"); s.set_defaults(fn=cmd_pick_ckpt)
    s = sp.add_parser("fit"); s.add_argument("run", nargs="?"); s.set_defaults(fn=cmd_fit)
    s = sp.add_parser("exclude-nodes"); s.add_argument("nodes", help="comma-separated node names")
    s.set_defaults(fn=cmd_exclude_nodes)
    s = sp.add_parser("prune"); s.add_argument("run"); s.add_argument("--apply", action="store_true")
    s.set_defaults(fn=cmd_prune)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
