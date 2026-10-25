"""Training on a cluster: data made while the GPU trains, several GPUs, jobs that end and resume.

* :class:`Producer` -- a background thread making training pairs on the training GPU on its own CUDA stream, so data
  generation overlaps training inside the one process (cluster GPUs are exclusive-process: no second process may use
  the card). Optional CPU-only worker processes (:class:`CpuWorkers`, never touching CUDA) add pairs from the job's
  spare cores. The training buffer takes whatever is ready each step and never waits.
* :func:`init_distributed` / :func:`allreduce_grads` -- data parallel over the GPUs of a node (``torchrun``): each rank
  its own data stream and buffer, gradients averaged after the step's backward passes (any number of forward passes
  per step, unlike DDP's hooks), rank 0 validates, logs and saves.
* :class:`Checkpointer` -- saves every N minutes and on the scheduler's warning signal (SGE ``-notify``: SIGUSR1 /
  SIGUSR2; SIGTERM), then exits with 99 so a wrapper resubmits; ``--resume auto`` picks the run up.
* :func:`load_partial` -- a warm start that also fills weights whose input channels grew (a new trunk level): the old
  channels copied, the new ones zero, so the network computes exactly what the old one did at the start.
"""
from __future__ import annotations

import os
import queue
import signal
import threading
import time

import torch


# ---------------------------------------------------------------------------------------------------------------
# data made while training
# ---------------------------------------------------------------------------------------------------------------

def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tensors(v)


class Producer:
    """Makes ``make(seed, size, device, strength, gen)`` pairs in a background thread on its own CUDA stream; ``cfg``
    (``strength``, ``gen``) is read at each pair, so a curriculum or a difficulty change reaches it at once."""

    def __init__(self, make, size: int, device, seed0: int, cfg: dict, maxq: int = 16):
        self.make, self.size, self.device, self.next, self.cfg = make, size, device, seed0, cfg
        self.q: queue.Queue = queue.Queue(maxq); self.stop = False; self.made = 0; self.t0 = time.time(); self.err = None
        self.stream = torch.cuda.Stream(device=device) if str(device).startswith("cuda") else None
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def _run(self):
        try:
            while not self.stop:
                seed = self.next; self.next += 1
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        item = self.make(seed, self.size, self.device, self.cfg["strength"], self.cfg["gen"])
                        ev = torch.cuda.Event(); ev.record(self.stream)
                else:
                    item = self.make(seed, self.size, self.device, self.cfg["strength"], self.cfg["gen"]); ev = None
                while not self.stop:
                    try:
                        self.q.put((item, ev), timeout=1.0); self.made += 1; break
                    except queue.Full:
                        continue
        except Exception as e:                                                                     # surfaced to the trainer
            self.err = e

    def take(self, k: int) -> list:
        """Up to ``k`` ready pairs (never waits); each made safe to use on the caller's stream."""
        if self.err is not None:
            raise RuntimeError(f"producer failed: {self.err!r}")
        out = []
        for _ in range(k):
            try:
                item, ev = self.q.get_nowait()
            except queue.Empty:
                break
            if ev is not None:
                cur = torch.cuda.current_stream(self.device)
                ev.wait(cur)
                for t in _tensors(item):
                    if t.is_cuda:
                        t.record_stream(cur)                                                      # memory not reused while the main stream still reads it
            out.append(item)
        return out

    def rate(self) -> float:
        return self.made / max(time.time() - self.t0, 1e-6)

    def close(self):
        """Stop and join the thread (it may be mid-pair: let it finish, so no torch op runs at interpreter exit)."""
        self.stop = True
        while self.th.is_alive():
            try:
                self.q.get_nowait()                                                                # unblock a pending put
            except queue.Empty:
                pass
            self.th.join(timeout=0.5)


def _cpu_worker(rank: int, n: int, size: int, seed0: int, cfg, q, state: dict, threads: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                                                         # never touch the (exclusive) GPU
    import torch as _t
    _t.set_num_threads(threads)
    from . import train as TR
    from . import losses as L
    for k, v in state.items():                                                                     # the trainer's target settings
        if k.startswith("L."):
            setattr(L, k[2:], v)
        else:
            setattr(TR, k, v)
    k = 0
    while True:
        # whole scenes per worker: blocks of ``crops_per_scene`` consecutive seeds (one scene each, rendered once and
        # cropped from the cache); the workers' blocks interleave -- every seed still drawn once
        cps = max(1, int(((cfg["gen"] or {}).get("scene") or {}).get("crops_per_scene", 1) or 1))
        seed = (seed0 // cps + (k // cps) * n + rank) * cps + k % cps; k += 1
        try:
            img, t = TR._make_pair(seed, size, "cpu", float(cfg["strength"]), dict(cfg["gen"]))
            q.put((img, t))
        except Exception as e:
            q.put(("error", repr(e))); return


class CpuWorkers:
    """``n`` CPU-only processes making pairs (``threads`` each) into a queue; ``cfg`` a shared dict (manager)."""

    def __init__(self, n: int, size: int, seed0: int, cfg: dict, state: dict, threads: int = 4, maxq: int = 32):
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.manager = ctx.Manager(); self.cfg = self.manager.dict(cfg); self.q = ctx.Queue(maxq)
        self.procs = [ctx.Process(target=_cpu_worker, args=(r, n, size, seed0, self.cfg, self.q, state, threads), daemon=True) for r in range(n)]
        for p in self.procs:
            p.start()
        self.made = 0; self.t0 = time.time()

    def update(self, **kw):
        for k, v in kw.items():
            self.cfg[k] = v

    def take(self, k: int, device) -> list:
        out = []
        for _ in range(k):
            try:
                img, t = self.q.get_nowait()
            except queue.Empty:
                break
            if isinstance(img, str):
                raise RuntimeError(f"cpu worker failed: {t}")
            out.append((img.to(device, non_blocking=True), {kk: vv.to(device, non_blocking=True) for kk, vv in t.items()})); self.made += 1
        return out

    def rate(self) -> float:
        return self.made / max(time.time() - self.t0, 1e-6)

    def close(self):
        for p in self.procs:
            p.terminate()


# ---------------------------------------------------------------------------------------------------------------
# several GPUs
# ---------------------------------------------------------------------------------------------------------------

def init_distributed(device: str = "cuda"):
    """``(rank, world, local_rank)`` from torchrun's environment; NCCL (GPUs) or gloo (CPU tests) when world > 1."""
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        if str(device).startswith("cuda"):
            torch.cuda.set_device(local); dist.init_process_group("nccl")
        else:
            dist.init_process_group("gloo")
    return rank, world, local


def allreduce_grads(model, world: int):
    """Average the gradients over the ranks (one flat all-reduce)."""
    if world <= 1:
        return
    import torch.distributed as dist
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat); flat /= world
    i = 0
    for g in grads:
        n = g.numel(); g.copy_(flat[i:i + n].view_as(g)); i += n


def allreduce_mean(x: torch.Tensor, world: int) -> torch.Tensor:
    if world <= 1:
        return x
    import torch.distributed as dist
    y = x.clone(); dist.all_reduce(y); return y / world


# ---------------------------------------------------------------------------------------------------------------
# jobs that end
# ---------------------------------------------------------------------------------------------------------------

class Checkpointer:
    """Save every ``minutes`` and when the scheduler warns (SIGUSR1 / SIGUSR2 / SIGTERM): the trainer polls
    :attr:`due` and :attr:`stop`; after a signal-triggered save the job exits with code 99 (requeue)."""

    def __init__(self, minutes: float = 20.0):
        self.every = minutes * 60; self.last = time.time(); self.stop = False
        for sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):                                                           # not the main thread
                pass

    def _on_signal(self, signum, frame):
        self.stop = True

    def due(self) -> bool:
        if self.stop or time.time() - self.last >= self.every:
            self.last = time.time(); return True
        return False


def load_partial(model, sd: dict, log=print) -> int:
    """Load ``sd`` into ``model`` where names match: equal shapes copied; a weight whose input channels (dim 1) grew
    gets the old channels and zeros for the new (so the start computes what the old network did); other mismatches
    keep their fresh initialisation. Returns the number of tensors loaded."""
    own = model.state_dict(); new = {}; n_grow = 0
    for k, v in sd.items():
        if k not in own:
            continue
        t = own[k]
        if t.shape == v.shape:
            new[k] = v
        elif t.dim() == v.dim() and t.dim() >= 2 and t.shape[0] == v.shape[0] and t.shape[1] > v.shape[1] and t.shape[2:] == v.shape[2:]:
            w = torch.zeros_like(t); w[:, :v.shape[1]] = v; new[k] = w; n_grow += 1
    model.load_state_dict(new, strict=False)
    log(f"init: {len(new)}/{len(own)} tensors loaded ({n_grow} with grown input channels)")
    return len(new)


def finish_distributed(world: int):
    if world > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier(); dist.destroy_process_group()
