"""Data-parallel QUESTION_CONDITIONED_WRITES encoder training: one process per GPU, launched by ``torch.distributed.run``.

Every rank loads the frozen backend, the cached latents, and the teacher responses, takes every world-size-th
question of each update, and all-reduces the token-weighted encoder gradients, so the update
equals the single-process one (DISTRACTOR_CURVES's scheme). Rank 0 writes checkpoints and the result JSON.
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
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=2))
    try:
        unit = json.loads(args.unit.read_text())
        os.environ["EXPERIMENT_STATE_DIR"] = str(args.side.parent)
        import question_conditioned_writes_runner as runner
        from question_conditioned_writes_conditioned_write import train

        context = runner.setup({"config": unit["plan_config"]}, f"ddp-rank{dist.get_rank()}")
        result = train(runner.backend(context), runner.data(context), runner.latents(context),
                       runner.responses(context), args.side / "encoders" / f"seed{unit['seed']}",
                       **runner.train_options(unit))
        dist.barrier()
        if dist.get_rank() == 0:
            args.output.write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
