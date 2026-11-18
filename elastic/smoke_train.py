"""Toy trainer for smoking the elastic machinery without a 5 GB model.

Same contract as scripts/train_lmdm.py: --batch_size (per GPU) / --accum_batches / --save_dir /
--ckpt_path / --steps / --checkpoint_every, ElasticCallback + slurm_topology, rolling
ModelCheckpoint. On top it counts the samples that actually went into every optimizer step
(all-reduced over ranks) and fails if that is not ELASTIC_GLOBAL_BS, so a resize that silently
changes the effective batch cannot pass. Each optimizer step's line is the evidence:

    [smoke] step 412 world 2 accum 2 samples 8 lr 9.6e-04 loss 0.0123

2026-11-17 (CSAIL step 0): ``--nccl_bench_mb N`` times an N MB bf16 all_reduce at train start and prints
algbw/busbw (rank 0) plus one ``[smoke] rank r/w host h gpu g`` line per rank; ``--hang_at_step N`` makes
the last rank sleep forever at step N once per save_dir (marker file), to prove a hung multi-node job is
cancelled by the controller's stall check and re-raced instead of eating a day.
"""

import argparse
import os
import socket
import sys
import time

import pytorch_lightning as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_audio_3.training.elastic import ElasticCallback, slurm_topology  # noqa: E402


class Toy(pl.LightningModule):
    def __init__(self, steps, sleep, bench_mb=0, hang_at=0, save_dir=""):
        super().__init__()
        self.bench_mb, self.hang_at, self.save_dir = bench_mb, hang_at, save_dir
        self.net = torch.nn.Sequential(torch.nn.Linear(32, 256), torch.nn.GELU(),
                                       torch.nn.Linear(256, 1))
        self.steps, self.sleep = steps, sleep
        self._seen = 0

    def on_train_start(self):
        import torch.distributed as dist
        r, w = self.trainer.global_rank, self.trainer.world_size
        dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
        name = torch.cuda.get_device_name(dev) if dev >= 0 else "cpu"
        print(f"[smoke] rank {r}/{w} host {socket.gethostname().split('.')[0]} local "
              f"{self.trainer.local_rank} gpu {dev} {name} CUDA_VISIBLE_DEVICES="
              f"{os.environ.get('CUDA_VISIBLE_DEVICES', '-')}", flush=True)
        if self.bench_mb and dist.is_available() and dist.is_initialized() and w > 1:
            n = self.bench_mb * 2**20 // 2
            t = torch.ones(n, dtype=torch.bfloat16, device=self.device)
            for _ in range(3):
                dist.all_reduce(t)
            torch.cuda.synchronize()
            iters = 10
            t0 = time.time()
            for _ in range(iters):
                dist.all_reduce(t)
            torch.cuda.synchronize()
            dt_s = (time.time() - t0) / iters
            alg = self.bench_mb / 1024 / dt_s
            if r == 0:
                print(f"[smoke] allreduce {self.bench_mb} MB bf16 world {w}: {dt_s * 1e3:.1f} ms "
                      f"algbw {alg:.2f} GB/s busbw {alg * 2 * (w - 1) / w:.2f} GB/s", flush=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.hang_at and self.trainer.global_step == self.hang_at \
                and self.trainer.global_rank == self.trainer.world_size - 1:
            mark = os.path.join(self.save_dir, f".hung_once_{self.hang_at}")
            if not os.path.exists(mark):
                open(mark, "w").close()
                print(f"[smoke] rank {self.trainer.global_rank} HANGING at step {self.hang_at} "
                      f"(test)", flush=True)
                while True:
                    time.sleep(3600)
        self._seen += x.shape[0]
        time.sleep(self.sleep)
        loss = torch.nn.functional.mse_loss(self.net(x), y)
        self._loss = loss.detach()
        return loss

    def on_before_optimizer_step(self, optimizer):
        seen = self.trainer.strategy.reduce(torch.tensor(float(self._seen), device=self.device),
                                            reduce_op="sum")
        self._seen = 0
        want = int(os.environ.get("ELASTIC_GLOBAL_BS", "0"))
        step = self.trainer.global_step + 1
        if self.trainer.is_global_zero and (step % 10 == 0 or step < 3):
            print(f"[smoke] step {step} world {self.trainer.world_size} "
                  f"accum {self.trainer.accumulate_grad_batches} samples {int(seen.item())} "
                  f"lr {optimizer.param_groups[0]['lr']:.3e} loss {self._loss.item():.4f} t {time.time():.2f}",
                  flush=True)
        if want and int(seen.item()) != want:
            raise SystemExit(f"[smoke] FATAL: optimizer step {step} saw {int(seen.item())} "
                             f"samples, expected global batch {want}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / self.steps))
        return [opt], [{"scheduler": sched, "interval": "step"}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--accum_batches", type=int, default=1)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--checkpoint_every", type=int, default=200)
    p.add_argument("--step_sleep", type=float, default=0.2)
    p.add_argument("--nccl_bench_mb", type=int, default=0)
    p.add_argument("--hang_at_step", type=int, default=0)
    args = p.parse_args()

    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 32, generator=g)
    y = (x[:, :1] * 2 + x[:, 1:2].sin()).contiguous()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                     batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     num_workers=2, persistent_workers=True)
    plugins, num_nodes = slurm_topology()
    trainer = pl.Trainer(
        devices="auto", accelerator="auto", num_nodes=num_nodes, plugins=plugins,
        precision="bf16-mixed", accumulate_grad_batches=args.accum_batches,
        max_steps=args.steps, max_epochs=-1, default_root_dir=args.save_dir, logger=False,
        enable_progress_bar=False, num_sanity_val_steps=0,
        callbacks=[ElasticCallback(args.save_dir),
                   pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every,
                                                dirpath=args.save_dir, save_top_k=1)])
    trainer.fit(Toy(args.steps, args.step_sleep, args.nccl_bench_mb, args.hang_at_step,
                    args.save_dir), dl, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
