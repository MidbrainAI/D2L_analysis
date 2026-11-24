"""Freeze and audit the immutable QA_PROTOCOL SQuAD/DROP/ROPES manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_protocol_protocol import freeze_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="FAMILY=JSONL")
    parser.add_argument("--chunk-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", action="append", required=True, metavar="FAMILY=REVISION")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--prompt-template-version", required=True)
    parser.add_argument("--min-validation-rows", type=int, default=134)
    parser.add_argument("--min-test-rows", type=int, default=1)
    args = parser.parse_args()
    manifest = freeze_manifest(
        args.source,
        args.chunk_audit,
        args.output,
        source_revisions=args.source_revision,
        tokenizer_revision=args.tokenizer_revision,
        prompt_template_version=args.prompt_template_version,
        min_validation_rows=args.min_validation_rows,
        min_test_rows=args.min_test_rows,
    )
    print(json.dumps({"status": "frozen", "manifest_sha256": manifest["manifest_sha256"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
