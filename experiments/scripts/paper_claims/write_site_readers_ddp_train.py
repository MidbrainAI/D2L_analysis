"""Data-parallel WRITE_SITE_READERS reader training, launched by ``torch.distributed.run`` (see distractor_curves_ddp_train)."""

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
    # 30 min: when one rank dies (e.g. OOM) the others must fail fast instead of billing a 2 h NCCL wait.
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(minutes=30))
    try:
        unit = json.loads(args.unit.read_text())
        import distractor_curves_runner as runner
        from distractor_curves_reader_training import load_latents
        from write_site_readers_reader_training import SELECTION_SUITE, train_reader

        os.environ["EXPERIMENT_STATE_DIR"] = str(args.side.parent)
        context = runner.setup({}, f"ddp-rank{dist.get_rank()}")
        backend = runner.backend_for(context, unit["model"])
        payload = runner.reader_manifest(unit["model"])
        latents = load_latents(args.side / "latents" / unit["model"])
        selection = json.loads((runner.ROOT / unit.get("selection_suite", str(SELECTION_SUITE))).read_text())
        result = train_reader(backend, payload, latents, selection,
                              args.side / "readers" / unit["model"] / f"{unit['arm']}-seed{unit['seed']}",
                              arm=unit["arm"], seed=unit["seed"], steps=unit["steps"],
                              max_document_occurrences=unit["max_document_occurrences"],
                              validation_interval=unit.get("validation_interval", 500),
                              validation_limit=unit.get("validation_limit"),
                              fresh_per_k=unit.get("fresh_per_k", 40))
        dist.barrier()
        if dist.get_rank() == 0:
            args.output.write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
