"""Data-parallel reader training: one process per GPU, launched by ``torch.distributed.run``.

Every rank loads the frozen backend on its own GPU and the cached latents, takes every
world-size-th question of each 12-question update, and all-reduces the token-weighted reader
gradients, so the update equals the single-process one. Rank 0 writes checkpoints and the
result JSON. The runner starts this for ``train`` units when the instance has several GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # gloo lets the same path run as a multi-process CPU parity check.
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=2))
    try:
        unit = json.loads(args.unit.read_text())
        import distractor_curves_runner as runner
        from distractor_curves_reader_training import load_latents, train_reader

        os.environ["EXPERIMENT_STATE_DIR"] = str(args.side.parent)
        context = runner.setup({}, f"ddp-rank{dist.get_rank()}")
        backend = runner.backend_for(context, unit["model"])
        payload = runner.reader_manifest(unit["model"])
        latents = load_latents(args.side / "latents" / unit["model"])
        result = train_reader(backend, payload, latents, args.side / "readers" / unit["model"] / f"seed{unit['seed']}",
                              seed=unit["seed"], steps=unit["steps"],
                              max_document_occurrences=unit["max_document_occurrences"],
                              validation_interval=unit.get("validation_interval", 500),
                              validation_limit=unit.get("validation_limit"))
        dist.barrier()
        if dist.get_rank() == 0:
            args.output.write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
