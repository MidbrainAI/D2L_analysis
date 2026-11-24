"""Materialize pinned SQuAD/DROP/ROPES source snapshots for QA_PROTOCOL.

This reuses the repository's existing Hugging Face preparation path.  It refuses
floating dataset revisions so the resulting JSONL files can be audited and
frozen later without changing the input corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import download_manifest, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parents[2]
    config = load_config(root, args.config)
    sources = [source for source in config["data"]["sources"] if source["dataset"] in {"squad", "drop", "ropes"}]
    if any(not source.get("revision") for source in sources):
        raise ValueError("QA_PROTOCOL source preparation requires an explicit revision on every SQuAD/DROP/ROPES source")
    config["paths"]["manifest"] = str(args.output / "combined-source.jsonl")
    combined = download_manifest(root, config)
    args.output.mkdir(parents=True, exist_ok=True)
    by_family = {family: [] for family in ("squad", "drop", "ropes")}
    for line in combined.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["dataset"] in by_family:
                by_family[row["dataset"]].append(row)
    for family, rows in by_family.items():
        rows.sort(key=lambda row: (str(row.get("split", "")), str(row.get("id", "")), str(row.get("document_id", ""))))
        target = args.output / f"{family}.jsonl"
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        content = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise ValueError(f"prepared source differs from existing immutable output: {target}")
        if not target.exists():
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(target)
    provenance_path = combined.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    provenance.update({
        "prepared_by": "qa_protocol_prepare_sources.py", "source_manifest": str(combined),
        "source_manifest_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
        "config": args.config,
        "family_files": {family: {"path": str(args.output / f"{family}.jsonl"),
                                  "sha256": hashlib.sha256((args.output / f"{family}.jsonl").read_bytes()).hexdigest(),
                                  "rows": len(rows)} for family, rows in by_family.items()},
    })
    provenance_target = args.output / "provenance.json"
    provenance_content = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if provenance_target.exists() and provenance_target.read_text(encoding="utf-8") != provenance_content:
        raise ValueError(f"prepared provenance differs from existing immutable output: {provenance_target}")
    if not provenance_target.exists():
        temporary = args.output / f".provenance.json.tmp-{os.getpid()}"
        temporary.write_text(provenance_content, encoding="utf-8")
        temporary.replace(provenance_target)
    print(json.dumps({"status": "prepared", "output": str(args.output), "source_manifest": str(combined),
                      "provenance": str(args.output / "provenance.json")}, indent=2))


if __name__ == "__main__":
    main()
