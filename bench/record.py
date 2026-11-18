#!/usr/bin/env python3
"""Append a single validated 26-column row to <site>/results.tsv."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

COLUMNS = [
    "trial",
    "commit",
    "started_jst",
    "decode_c1_tps",
    "total_c4_tps",
    "total_c8_tps",
    "total_c16_tps",
    "ttft_c1_s",
    "prefill_tps",
    "gpqa_set",
    "gpqa_correct",
    "gpqa_total",
    "hash_match",
    "peak_mem_gib",
    "status",
    "description",
    "engine",
    "engine_rev",
    "quant",
    "weights_rev",
    "parent",
    "gain_vs_parent_pct",
    "wr_ref",
    "vs_wr_pct",
    "agent",
    "cu_spent",
]


def clean_cell(val: object) -> str:
    if val is None:
        return ""
    s = str(val).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    return s


def ensure_header(tsv_path: pathlib.Path) -> list[dict[str, str]]:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        tsv_path.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
        return []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Header mismatch in {tsv_path}: {reader.fieldnames}")
        return list(reader)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", type=pathlib.Path, required=True, help="Target results.tsv path")
    ap.add_argument("--row-json", type=pathlib.Path, help="JSON file containing column values")
    for col in COLUMNS:
        ap.add_argument(f"--{col.replace('_', '-')}", dest=col, default=None)
    args = ap.parse_args()

    existing = ensure_header(args.tsv)
    payload: dict[str, object] = {}
    if args.row_json:
        payload.update(json.loads(args.row_json.read_text(encoding="utf-8")))
    for col in COLUMNS:
        cli_val = getattr(args, col)
        if cli_val is not None:
            payload[col] = cli_val

    status = clean_cell(payload.get("status"))
    if status not in {"keep", "discard", "crash"}:
        raise ValueError(f"Invalid status: {status!r}")

    if status == "crash":
        for metric in ("decode_c1_tps", "total_c4_tps", "total_c8_tps", "total_c16_tps", "ttft_c1_s", "prefill_tps", "peak_mem_gib"):
            if not clean_cell(payload.get(metric)):
                payload[metric] = "0"

    if payload.get("trial") is None or clean_cell(payload.get("trial")) == "":
        next_trial = 0 if not existing else max(int(r["trial"]) for r in existing) + 1
        payload["trial"] = str(next_trial)

    line = "\t".join(clean_cell(payload.get(col)) for col in COLUMNS)
    if len(line.split("\t")) != 26:
        raise AssertionError("Row did not serialize to 26 tab-separated columns")

    with args.tsv.open("a", encoding="utf-8", newline="") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
