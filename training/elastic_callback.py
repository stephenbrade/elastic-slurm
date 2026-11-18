"""Preemption-safe, resizable training on Slurm (Blue Vela). Pairs with ``elastic/``.

Two pieces live inside the training process:

* ``ElasticCallback`` turns scheduler signals into checkpoints. Blue Vela preempts with
  ``PreemptMode=REQUEUE`` and a 10 min grace, and ``send_user_signal`` delivers the job's
  ``--signal`` at the start of that grace. ``elastic/job.sbatch`` asks for ``B:USR1`` and forwards
  it to this process:

    - SIGUSR1: save ``elastic-step=N.ckpt`` at the next optimizer-step boundary, keep training
      (the scheduler kills us at the end of the grace; the requeued job resumes from the save).
      The signal comes at the START of the 10 min grace, so steps trained after that save died with
      the job (2026-10-27: ~35 per preemption). After USR1 the callback therefore keeps saving every
      ``ELASTIC_GRACE_SAVE_EVERY_S`` (default 240 s) until it is killed.
    - Time-based rolling saves every ``ELASTIC_SAVE_EVERY_S`` (0 = off; the launcher sets 600): with
      500-step checkpoints a preemption on 4 GPUs could cost ~2.5 h of training.
    - SIGUSR2: save, then stop cleanly. The controller sends this to hand a run over to a job
      with a different GPU count.

  Only the process that received the signal knows about it (under the subprocess launcher that
  is rank 0 alone), so the decision is all-reduced at every optimizer step. Every rank must
  reach ``trainer.save_checkpoint`` on the same step or the barrier inside it deadlocks.

* ``slurm_topology()`` picks the cluster environment. One node keeps the historical
  ``LightningEnvironment`` (see the comment in ``scripts/train_lmdm.py``); a multi-node job is
  launched with ``srun`` (one task per GPU) and needs ``SLURMEnvironment`` plus ``num_nodes``.

The elastic launcher also exports ``ELASTIC_GLOBAL_BS`` and ``ELASTIC_MICRO_BS``; the callback
checks ``micro x world x accum == global`` and the first batch's size, and refuses to train if
the effective batch the launcher promised is not the one Lightning is running.

NaN / corruption guards (2026-11-12, after ``ss_bc1_lag4_jam_mix2stem_cenc`` resumed from a NaN
checkpoint at 22k and trained and checkpointed NaN to 100k without anything stopping it):

* **Non-finite loss stop.** ``ElasticCallback`` ORs ``~isfinite(loss)`` over the micro-batches of
  each optimizer step (on device, no extra sync) and all-reduces it in the SAME collective as the
  save decision, so every rank agrees. After ``NONFINITE_MAX_STEPS`` (default 5; 0 = off)
  consecutive optimizer steps with a non-finite loss it records a ``nonfinite`` event plus a
  ``<run>/NONFINITE`` marker and raises ``NonFiniteError`` on every rank: non-zero exit, and the
  controller halts the run instead of requeueing it. No save is taken while losses are non-finite.
* **No non-finite checkpoint, ever.** ``ElasticCallback.on_save_checkpoint`` runs inside every
  ``trainer.save_checkpoint`` (routine, milestone, elastic/emergency) on every rank before anything
  is written, and raises ``NonFiniteError`` if a trainable parameter has NaN/inf or any
  ``state_dict`` tensor has NaN. (``ELASTIC_SAVE_FINITE_CHECK=0`` disables it.)
* **Atomic writes.** ``AtomicCheckpointIO`` (returned by ``slurm_topology()``, so every trainer that
  uses it gets it) writes ``<name>.tmp`` then ``os.replace``s it onto the final name; a save killed
  mid-write leaves only a ``.tmp``, which nothing resumes from or prunes as a checkpoint.
* Test only: ``ELASTIC_TEST_NAN_AT_STEP=N`` (unset by default) poisons one trainable parameter
  with NaN at the first batch at or after global step N, to exercise the guards end to end.
"""

import json
import os
import re
import signal
import time

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment

EMERGENCY_PREFIX = "elastic-"
NONFINITE_MARKER = "NONFINITE"        # <run>/NONFINITE; elastic.py's controller halts on it


class NonFiniteError(RuntimeError):
    """Training produced (or was about to save) non-finite values. The controller halts on it."""


def _rank():
    for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if os.environ.get(k):
            return int(os.environ[k])
    return 0


def elastic_event(kind, **kw):
    """Append to ``$ELASTIC_RUN/events.jsonl`` (same record format as elastic/elastic.py). No-op
    outside an elastic job; never raises."""
    rd = os.environ.get("ELASTIC_RUN")
    if not rd:
        return
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind,
           "job": os.environ.get("SLURM_JOB_ID"), **kw}
    try:
        with open(os.path.join(rd, "events.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def mark_nonfinite(where, detail):
    """Tell the elastic controller this job died of non-finite values: an event and the
    ``<run>/NONFINITE`` marker (job id + reason). The controller halts the run and notifies;
    ``elastic.py resume`` clears the marker. No-op outside an elastic job; never raises."""
    rd = os.environ.get("ELASTIC_RUN")
    if not rd:
        return
    elastic_event("nonfinite", where=where, detail=str(detail)[:1000], rank=_rank())
    rec = {"job": os.environ.get("SLURM_JOB_ID"), "where": where, "detail": str(detail)[:1000],
           "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "rank": _rank()}
    try:
        path = os.path.join(rd, NONFINITE_MARKER)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


class AtomicCheckpointIO(TorchCheckpointIO):
    """Write ``<path>.tmp``, fsync, then ``os.replace`` onto ``<path>`` (a file rename is atomic on
    Blue Vela's GPFS; directory renames fail there, so none is used). Lightning's own
    ``_atomic_save`` is NOT atomic here: its fsspec transaction stages in node-local /tmp and the
    commit is a cross-device copy straight into the final name. A path that already ends in
    ``.tmp`` (ElasticCallback's own staging name) is written as is."""

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = str(path)
        if path.endswith(".tmp"):
            return super().save_checkpoint(checkpoint, path, storage_options)
        tmp = path + ".tmp"
        super().save_checkpoint(checkpoint, tmp, storage_options)
        try:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        os.replace(tmp, path)


def _any_flag(tensors, fn):
    """True if ``fn(chunk)`` (a 0-d bool tensor) holds for any 16M-element chunk of any tensor.
    One host sync per device; chunking keeps the temporary mask small on a full GPU."""
    flags = {}
    for t in tensors:
        for c in t.detach().reshape(-1).split(1 << 24):
            flags.setdefault(c.device, []).append(fn(c))
    return any(bool(torch.stack(v).any()) for v in flags.values())


def _nonfinite(c):
    return ~torch.isfinite(c).all()


def _has_nan(c):
    return torch.isnan(c).any()


def nonfinite_report(pl_module, state_dict=None, limit=8):
    """Names of trainable parameters with NaN/inf and of ``state_dict`` tensors with NaN (buffers
    may legitimately hold -inf, e.g. attention masks; NaN is never legitimate). [] = clean."""
    params = [(n, p) for n, p in pl_module.named_parameters()
              if p.requires_grad and p.is_floating_point()]
    sd = [(k, v) for k, v in (state_dict or {}).items()
          if torch.is_tensor(v) and v.is_floating_point()]
    if not _any_flag([p for _, p in params], _nonfinite) and not _any_flag([v for _, v in sd], _has_nan):
        return []
    bad = [n for n, p in params if _any_flag([p], _nonfinite)][:limit]
    bad += [f"{k} (state_dict NaN)" for k, v in sd if _any_flag([v], _has_nan)][:limit]
    return bad or ["<unlocated non-finite value>"]


def _loss_of(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict) and torch.is_tensor(outputs.get("loss")):
        return outputs["loss"]
    return None


def slurm_topology():
    """(plugins, num_nodes) for ``pl.Trainer``.

    ``SLURM_NTASKS > 1`` means ``elastic/job.sbatch`` started us under ``srun`` with one task per
    GPU, so Slurm owns the topology. ``auto_requeue=False`` because Lightning's own SIGUSR1
    handler would otherwise save an hpc checkpoint and ``scontrol requeue`` the job itself,
    racing ``ElasticCallback`` and the controller.

    Every shape also gets ``AtomicCheckpointIO`` (2026-11-12): routine and milestone checkpoints
    are written to ``<name>.tmp`` and renamed into place, so a kill mid-save never leaves a
    truncated file under a resumable name.
    """
    io = AtomicCheckpointIO()
    if not SLURMEnvironment.detect():
        return [io], 1
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        return [SLURMEnvironment(auto_requeue=False), io], int(os.environ.get("SLURM_NNODES", "1"))
    return [LightningEnvironment(), io], 1


def pin_local_cuda_device():
    """Make bare ``"cuda"`` mean this rank's GPU. Call before building any model.

    Lightning's single-node launcher re-runs the script once per rank but only calls
    ``set_device`` inside ``trainer.fit``. Anything moved to ``"cuda"`` earlier lands on GPU 0 and
    its cached blocks stay there after Lightning moves the module: at 1x8 SA3-medium that was
    7 x 8 GiB on GPU 0, and the rank-0 optimizer restore OOM'd on 80 GB H100s (2026-10-26).
    """
    if not torch.cuda.is_available():
        return
    local = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    torch.cuda.set_device(int(local))


def _first_batch_dim(batch):
    if torch.is_tensor(batch):
        return batch.shape[0]
    if isinstance(batch, (list, tuple)):
        for b in batch:
            n = _first_batch_dim(b)
            if n is not None:
                return n
    if isinstance(batch, dict):
        for b in batch.values():
            n = _first_batch_dim(b)
            if n is not None:
                return n
    return None


class ResumeReseed(pl.Callback):
    """Make a resumed run draw NEW data instead of replaying the start of its epoch.

    Lightning restores ``current_epoch`` on resume but restarts that epoch from batch 0, and
    ``seed_everything`` re-seeds the main-process RNGs identically, so every restart re-drew the
    same batch order (stratified sampler) and the same random crops (Python ``random`` in each
    worker is seeded from the main torch generator). With a preemption every hour or so a run
    could see the same first ~hour of its epoch dozens of times (2026-10-28). On resume
    (``global_step > 0``) this folds the step into the sampler's RNG and re-seeds the main
    process RNGs with ``seed + global_step``; a fresh run is untouched. It runs in
    ``on_train_start``, before Lightning builds the dataloader iterator.
    """

    def __init__(self, seed, batch_sampler=None):
        super().__init__()
        self.seed = int(seed)
        self.batch_sampler = batch_sampler

    def on_train_start(self, trainer, pl_module):
        step = int(trainer.global_step)
        if step <= 0:
            return
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "resume_step"):
            self.batch_sampler.resume_step = step
        import random
        import numpy as np
        s = (self.seed + step) % (2 ** 31)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        print(f"[elastic] resumed at step {step}: data RNG re-seeded with seed+step={s}", flush=True)


class ElasticCallback(pl.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self._requested = None        # None | "save" | "stop"; set by the signal handler
        self._last_step = None
        self._every_s = float(os.environ.get("ELASTIC_SAVE_EVERY_S", "0") or 0)
        self._grace_every_s = float(os.environ.get("ELASTIC_GRACE_SAVE_EVERY_S", "240") or 240)
        self._last_save_t = time.time()
        self._checked_batch = False
        # NaN guard (2026-11-12): consecutive optimizer steps with a non-finite loss before the job
        # fails. Any NaN loss poisons the weights through backward (bf16, no grad scaler), so a
        # healthy run never sees one; 5 only rides out a single odd batch report.
        self._nf_max = int(os.environ.get("NONFINITE_MAX_STEPS", "5") or 0)
        self._nf_local = None         # device bool: a non-finite loss in this optimizer step so far
        self._nf_run = 0              # consecutive optimizer steps with a non-finite loss
        self._save_check = os.environ.get("ELASTIC_SAVE_FINITE_CHECK", "1") != "0"
        self._marked = False
        self._inject_at = int(os.environ.get("ELASTIC_TEST_NAN_AT_STEP", "0") or 0)   # TEST ONLY
        self._injected = False
        # Installed here, not in setup(): the signal can arrive while the model is still
        # loading, and SIGUSR1's default action kills the process. Each spawned rank re-runs
        # the script, so it constructs (and installs) its own copy.
        self._install()

    def _install(self):
        signal.signal(signal.SIGUSR1, self._on_signal)
        signal.signal(signal.SIGUSR2, self._on_signal)

    def _on_signal(self, signum, _frame):
        want = "stop" if signum == signal.SIGUSR2 else "save"
        if self._requested != "stop":
            self._requested = want
        if signum == signal.SIGUSR1 and self._grace_every_s > 0:
            # in the preemption grace now: keep the at-risk window short until the kill
            self._every_s = min(self._every_s or self._grace_every_s, self._grace_every_s)
        print(f"[elastic] received {signal.Signals(signum).name} -> {want} at next step boundary",
              flush=True)

    def setup(self, trainer, pl_module, stage):
        # Lightning's signal connector registers its handlers before callback setup; make sure
        # ours still own USR1/USR2.
        self._install()

    def on_train_start(self, trainer, pl_module):
        self._last_step = trainer.global_step
        world = trainer.world_size
        accum = getattr(trainer, "accumulate_grad_batches", 1)
        if not pl_module.automatic_optimization:
            accum = getattr(pl_module, "gradient_accumulation_steps", 1)
        micro = int(os.environ.get("ELASTIC_MICRO_BS", "0"))
        want = int(os.environ.get("ELASTIC_GLOBAL_BS", "0"))
        if trainer.is_global_zero:
            print(f"[elastic] world={world} (nodes={trainer.num_nodes}) micro_bs={micro or '?'} "
                  f"accum={accum} -> global_bs={micro * world * accum if micro else '?'} "
                  f"(target {want or 'unset'}); resuming at global_step={trainer.global_step}",
                  flush=True)
        if micro and want and micro * world * accum != want:
            raise SystemExit(f"[elastic] FATAL: micro {micro} x world {world} x accum {accum} "
                             f"= {micro * world * accum} != target global batch {want}")
        self._align_accumulation(trainer, accum)

    @staticmethod
    def _align_accumulation(trainer, accum):
        """Resize resume fix (2026-11-17, CSAIL step 0). Lightning steps the optimizer when the
        in-epoch batch counter is a multiple of accumulate_grad_batches. s0_race2 saved at 1x4 (accum 2)
        and resumed at 1x1 (accum 8): the restored counter was 6 mod 8, so the first optimizer step
        came after 2 micro-batches, 32 of 128 samples (smoke_train's per-step check failed). Round the
        counter UP to the next multiple of the new accum: the first window is full again; the epoch
        just ends up to accum-1 micro-batches early. No effect when the counter is already aligned
        (every same-shape resume)."""
        loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        cur = getattr(getattr(loop, "batch_progress", None), "current", None)
        if cur is None or accum <= 1 or cur.ready % accum == 0:
            return
        pad = accum - cur.ready % accum
        for f in ("ready", "started", "processed", "completed"):
            setattr(cur, f, getattr(cur, f) + pad)
        if trainer.is_global_zero:
            print(f"[elastic] in-epoch batch counter {cur.ready - pad} not a multiple of accum {accum}: "
                  f"advanced by {pad} so the first optimizer step after this resize is a full window",
                  flush=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._inject_at > 0 and not self._injected and trainer.global_step >= self._inject_at:
            # TEST ONLY (ELASTIC_TEST_NAN_AT_STEP, unset by default): poison one trainable
            # parameter on every rank so the next forward is NaN, as a corrupted resume would be.
            self._injected = True
            for n, p in pl_module.named_parameters():
                if p.requires_grad and p.is_floating_point():
                    with torch.no_grad():
                        p.fill_(float("nan"))
                    print(f"[elastic] TEST: ELASTIC_TEST_NAN_AT_STEP={self._inject_at}: filled {n} "
                          f"with NaN at global_step {trainer.global_step}", flush=True)
                    break
        if self._checked_batch:
            return
        self._checked_batch = True
        micro = int(os.environ.get("ELASTIC_MICRO_BS", "0"))
        got = _first_batch_dim(batch)
        if micro and got is not None and got != micro:
            raise SystemExit(f"[elastic] FATAL: first batch has {got} items per rank, launcher "
                             f"promised micro_bs={micro}; the effective batch would be wrong")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Decide only where an optimizer step just completed, so a save never lands inside an
        # accumulation window. global_step advances identically on every rank, so every rank
        # takes the collective below on the same batch.
        if self._nf_max > 0:
            loss = _loss_of(outputs)
            if loss is not None:          # device op, no host sync; read in the collective below
                bad = ~torch.isfinite(loss.detach()).all()
                self._nf_local = bad if self._nf_local is None else (self._nf_local | bad)
        if trainer.global_step == self._last_step:
            return
        self._last_step = trainer.global_step
        code = {None: 0, "save": 1, "stop": 2}[self._requested]
        if code == 0 and self._every_s > 0 and time.time() - self._last_save_t >= self._every_s:
            code = 1                      # time-based rolling save; ranks agree via the max-reduce
        # [save code, non-finite flag] in ONE max-reduce: the same single collective + host sync per
        # optimizer step as before the NaN guard, and every rank reaches the same verdict.
        t = torch.zeros(2, dtype=torch.long, device=pl_module.device)
        t[0] = code
        if self._nf_local is not None:
            t[1] = self._nf_local.to(device=t.device, dtype=torch.long)
        self._nf_local = None
        t = trainer.strategy.reduce(t, reduce_op="max")
        code, nf = (int(x) for x in t.tolist())
        if self._nf_max > 0:
            self._nf_run = self._nf_run + 1 if nf else 0
            if nf and trainer.is_global_zero:
                print(f"[elastic] NON-FINITE loss at global_step {trainer.global_step} "
                      f"({self._nf_run}/{self._nf_max} in a row)", flush=True)
            if self._nf_run >= self._nf_max:
                msg = (f"loss non-finite for {self._nf_run} consecutive optimizer steps (through "
                       f"global_step {trainer.global_step}); stopping without saving. Resume only "
                       f"from a checkpoint that passes scripts/check_ckpt_finite.py.")
                if trainer.is_global_zero:
                    print(f"[elastic] FATAL: {msg}", flush=True)
                    mark_nonfinite("loss", msg)
                self._marked = True
                raise NonFiniteError(msg)
            if nf and code:
                # never checkpoint while the loss is non-finite: keep the request for later
                self._requested = {1: "save", 2: "stop"}[code] if self._requested != "stop" else "stop"
                return
        if code == 0:
            return
        self._requested = None
        self.save(trainer)
        if code == 2:
            if trainer.is_global_zero:
                print("[elastic] stop requested: exiting after checkpoint", flush=True)
            trainer.should_stop = True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Runs inside EVERY trainer.save_checkpoint (routine, milestone, elastic) on every rank,
        before anything is written: refuse to persist non-finite weights. DDP ranks hold identical
        weights, so all ranks raise together and nothing deadlocks in the save barrier."""
        if not self._save_check:
            return
        bad = nonfinite_report(pl_module, checkpoint.get("state_dict"))
        if not bad:
            return
        msg = (f"refusing to save a checkpoint at global_step {trainer.global_step}: non-finite "
               f"values in {', '.join(bad)}")
        if trainer.is_global_zero:
            print(f"[elastic] FATAL: {msg}", flush=True)
            mark_nonfinite("save", msg)
        self._marked = True
        raise NonFiniteError(msg)

    def on_exception(self, trainer, pl_module, exception):
        # A non-finite stop raised elsewhere (e.g. step_sync_dmd._finish, on one rank only) must
        # still reach the controller as `nonfinite`, or it would be retried as a plain crash.
        if self._marked:
            return
        if isinstance(exception, NonFiniteError) or "non-finite" in str(exception).lower():
            self._marked = True
            mark_nonfinite("exception", f"{type(exception).__name__}: {exception}")

    def save(self, trainer):
        self._last_save_t = time.time()
        step = trainer.global_step
        final = os.path.join(self.save_dir, f"{EMERGENCY_PREFIX}step={step}.ckpt")
        tmp = final + ".tmp"
        t0 = time.time()
        trainer.save_checkpoint(tmp)          # collective: rank 0 writes, all ranks barrier
        if trainer.is_global_zero:
            os.replace(tmp, final)            # file rename is atomic on GPFS (dir renames fail)
            for f in os.listdir(self.save_dir):
                m = re.fullmatch(rf"{EMERGENCY_PREFIX}step=(\d+)\.ckpt", f)
                if m and int(m.group(1)) < step:
                    os.remove(os.path.join(self.save_dir, f))
            print(f"[elastic] saved {final} in {time.time() - t0:.1f}s", flush=True)
        trainer.strategy.barrier("elastic-save")
