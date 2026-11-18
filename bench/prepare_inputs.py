#!/usr/bin/env python3
"""Fetch and deterministically assemble benchmark & GPQA Diamond guard inputs.

Public repository safe: never stores GPQA question text, choices, or answers inside
tracked repository paths. Writes the assembled private bundle to ../private/eval-bundle.json
(or --out outside tracked paths) and verifies SHA-256 digests against bench/guard-ids.sha256.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import pathlib
import random
import re
import subprocess
import urllib.request

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
DEFAULT_OUT = REPO_DIR.parent / "private" / "eval-bundle.json"
DEFAULT_CACHE = REPO_DIR.parent / "private" / "source-cache"

SOURCES = {
    "llmPrompts.js": (
        "https://raw.githubusercontent.com/MiaAI-Lab/sparkDash/main/src/shared/llmPrompts.js",
        "4938e3245d8474f27055bbc66493163e99f1b54885cd5a62a7c7b6507a208bee",
    ),
    "common.py": (
        "https://raw.githubusercontent.com/openai/simple-evals/main/common.py",
        "84616c52821a30ae3f55f153a763153fd833cb41b547b09f547b46e6337f8f35",
    ),
    "gpqa.csv": (
        "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
        "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
    ),
}

SPEED_BENCH_PARQUET = (
    "https://huggingface.co/datasets/nvidia/SPEED-Bench/resolve/"
    "fa4e669bee2a4393c56aa8ed05a7c667897f9590/qualitative/test-00000-of-00001.parquet",
    "bb5ddc59123d0d3aebfb5bd70654dd6b56dde00a44b27153f408616073d08afb",
)

CONSTITUTION_SHA256 = "032c395bc17e086ed532fed704c48eeafdaca13641f648f30d9a2f7fc83b82e7"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fetch_or_cache(name: str, url: str, expected_sha: str, cache_dir: pathlib.Path, strict_sha: bool) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if target.exists():
        raw = target.read_bytes()
        if not expected_sha or sha256_bytes(raw) == expected_sha or not strict_sha:
            return raw
    req = urllib.request.Request(url, headers={"User-Agent": "qfn-autoresearch-bench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    if strict_sha and expected_sha and sha256_bytes(raw) != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {name}: got {sha256_bytes(raw)}, expected {expected_sha}")
    target.write_bytes(raw)
    return raw


def parse_code_prompts(js_text: str) -> tuple[list[str], str]:
    def js_lines(key: str) -> str:
        block = js_text.split(f"{key} =", 1)[1].split(";\n", 1)[0]
        return "".join(json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', block))

    tail = js_lines("CODE_TASK_TAIL")
    code = [
        f"{name}\n{spec}\n{tail}"
        for name, spec in re.findall(r'codeTask\("([^"]+)", "([^"]+)"\)', js_text)
    ]
    prose = js_lines("DECODE_PROSE_PROMPT")
    if len(code) < 16 or len(tail) <= 80 or len(prose) <= 80:
        raise ValueError("Unexpected prompt extraction result from llmPrompts.js")
    return code[:16], prose


def parse_multichoice_template(common_py_text: str) -> str:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        module = ast.parse(common_py_text)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "QUERY_TEMPLATE_MULTICHOICE" for t in node.targets):
            return node.value.func.value.value.strip()
    raise ValueError("QUERY_TEMPLATE_MULTICHOICE not found in common.py")


def build_gpqa_splits(gpqa_csv_text: str, template: str) -> dict[str, list[dict]]:
    g198: list[dict] = []
    g13: list[dict] = []
    rng = random.Random(0)
    reader = csv.DictReader(io.StringIO(gpqa_csv_text))
    for i, row in enumerate(reader):
        order = rng.sample(range(4), 4)
        choices = [row["Correct Answer"]] + [row[f"Incorrect Answer {j}"] for j in (1, 2, 3)]
        rec_id = row["Record ID"]
        item = {
            "id": rec_id,
            "id_sha256": sha256_text(rec_id),
            "index": i,
            "subdomain": row["Subdomain"],
            "answer": "ABCD"[order.index(0)],
            "prompt": template.format(
                Question=row["Question"],
                **{k: choices[j] for k, j in zip("ABCD", order)},
            ),
        }
        g198.append(item)
        if row["Subdomain"] == "Astrophysics":
            g13.append(item)

    if len(g198) != 198 or len(g13) != 13:
        raise ValueError(f"Unexpected GPQA counts: g198={len(g198)}, g13={len(g13)}")

    g13_ids = {x["id"] for x in g13}
    rest = [x for x in g198 if x["id"] not in g13_ids]
    extra27 = random.Random(0).sample(rest, 27)
    g40 = sorted(g13 + extra27, key=lambda x: x["index"])
    return {"G13": g13, "G40": g40, "G198": g198}


def format_manifest(code_prompts: list[str], constitution_sha: str, splits: dict[str, list[dict]]) -> str:
    lines = [
        "# Deterministic SHA-256 manifest for prompts and GPQA Diamond guard record IDs",
        f"constitution\t0\t{constitution_sha}",
    ]
    for idx, prompt in enumerate(code_prompts):
        lines.append(f"code\t{idx}\t{sha256_text(prompt)}")
    for split_name in ("G13", "G40", "G198"):
        for item in splits[split_name]:
            lines.append(f"{split_name}\t{item['index']}\t{item['id_sha256']}")
    return "\n".join(lines) + "\n"


def ensure_not_tracked(out_path: pathlib.Path) -> None:
    resolved = out_path.resolve()
    try:
        rel = resolved.relative_to(REPO_DIR.resolve())
    except ValueError:
        return
    if rel.parts and rel.parts[0] == "private":
        return
    res = subprocess.run(
        ["git", "-C", str(REPO_DIR), "check-ignore", "-q", str(resolved)],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Refusing to write private GPQA bundle to non-gitignored repo path: {resolved}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT, help="Output bundle path (must be outside tracked repo)")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE, help="Directory for downloaded raw source files")
    ap.add_argument("--write-manifest", action="store_true", help="Write bench/guard-ids.sha256 instead of verifying")
    ap.add_argument("--no-strict-source-sha", action="store_true", help="Skip raw source file SHA check (still verifies prompt/ID manifest)")
    ap.add_argument("--with-speed-bench", action="store_true", help="Also prepare SPEED-Bench coding 8-conversation JSONL if pyarrow is installed")
    args = ap.parse_args()

    ensure_not_tracked(args.out)
    strict = not args.no_strict_source_sha

    const_path = BENCH_DIR / "constitution.txt"
    const_sha = sha256_bytes(const_path.read_bytes())
    if const_sha != CONSTITUTION_SHA256:
        raise ValueError(f"constitution.txt SHA-256 mismatch: {const_sha}")

    js_text = fetch_or_cache("llmPrompts.js", *SOURCES["llmPrompts.js"], args.cache_dir, strict).decode("utf-8")
    common_text = fetch_or_cache("common.py", *SOURCES["common.py"], args.cache_dir, strict).decode("utf-8")
    gpqa_text = fetch_or_cache("gpqa.csv", *SOURCES["gpqa.csv"], args.cache_dir, strict).decode("utf-8")

    code_prompts, prose_prompt = parse_code_prompts(js_text)
    template = parse_multichoice_template(common_text)
    splits = build_gpqa_splits(gpqa_text, template)

    manifest_text = format_manifest(code_prompts, const_sha, splits)
    manifest_path = BENCH_DIR / "guard-ids.sha256"
    if args.write_manifest:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    else:
        expected = manifest_path.read_text(encoding="utf-8")
        if manifest_text != expected:
            raise ValueError("Generated prompt/GPQA manifest does not match frozen bench/guard-ids.sha256")

    bundle = {
        "code": code_prompts,
        "prose": prose_prompt,
        "quality": splits["G13"],
        "gpqa_splits": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    if args.with_speed_bench:
        try:
            import pyarrow.parquet as pq  # type: ignore
            parquet_bytes = fetch_or_cache("speed.parquet", *SPEED_BENCH_PARQUET, args.cache_dir, strict)
            rows = [x for x in pq.read_table(io.BytesIO(parquet_bytes)).to_pylist() if x["category"] == "coding"]
            records = []
            for row in rows[:8]:
                records.append({
                    "question_id": row["question_id"],
                    "category": "coding",
                    "messages": [{"role": "user", "content": t} for t in row["turns"]],
                })
            if sum(len(r["messages"]) for r in records) != 10:
                raise ValueError("Expected 10 total turns across first 8 SPEED-Bench coding rows")
            speed_out = args.out.parent / "nvidia-coding8.jsonl"
            speed_out.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
        except ImportError:
            print("pyarrow not installed; skipping optional nvidia-coding8.jsonl generation")

    print(
        json.dumps(
            {
                "status": "ok",
                "code_prompts": len(code_prompts),
                "G13": len(splits["G13"]),
                "G40": len(splits["G40"]),
                "G198": len(splits["G198"]),
                "manifest_sha256": sha256_text(manifest_text),
            }
        )
    )


if __name__ == "__main__":
    main()
