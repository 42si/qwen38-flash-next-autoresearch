#!/usr/bin/env python3
"""GPQA Diamond accuracy guard (G13 / G40 / G198) and greedy output hash check.

Public repository safe: emits ONLY aggregate counts and per-item Record ID SHA-256
pass/fail booleans. Never writes question text, choices, explanations, or model responses.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

ANSWER_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
JST = dt.timezone(dt.timedelta(hours=9))


def detect_model(base_url: str, requested: str | None) -> str:
    if requested:
        return requested
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
        if "Qwen3.8-Flash-Next" in ids:
            return "Qwen3.8-Flash-Next"
        if ids:
            return ids[0]
    except Exception:
        pass
    return "Qwen3.8-Flash-Next"


def eval_one(base_url: str, model: str, item: dict, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True},
        "seed": 42,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    content: list[str] = []
    usage: dict = {}
    finish: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                d = json.loads(raw)
                if d.get("error"):
                    raise ValueError(str(d["error"])[:250])
                usage = d.get("usage") or usage
                for choice in d.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        content.append(delta["content"])
                    finish = choice.get("finish_reason") or finish
    except urllib.error.HTTPError as ex:
        raise RuntimeError(f"HTTP {ex.code} " + ex.read().decode("utf-8", errors="replace")[:400]) from ex

    msg = "".join(content)
    matches = ANSWER_RE.findall(msg)
    pred = matches[-1].upper() if matches else None
    id_sha = item.get("id_sha256") or hashlib.sha256(item["id"].encode("utf-8")).hexdigest()
    return {
        "id_sha256": id_sha,
        "index": item["index"],
        "correct": bool(pred == item["answer"]),
        "finish": finish,
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["check-hash", "eval"], required=True)
    ap.add_argument("--speed-summary", type=pathlib.Path, help="Candidate speed summary JSON (for check-hash)")
    ap.add_argument("--parent-hashes", type=pathlib.Path, help="Parent 16-prompt SHA-256 JSON (for check-hash)")
    ap.add_argument("--url", default="http://127.0.0.1:29771")
    ap.add_argument("--model", default=None)
    ap.add_argument("--bundle", type=pathlib.Path, help="Private eval-bundle.json (for eval)")
    ap.add_argument("--split", choices=["G13", "G40", "G198"], default="G40")
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    if args.mode == "check-hash":
        if not args.speed_summary or not args.parent_hashes or not args.parent_hashes.exists():
            result = {"hash_match": 0, "matched": 0, "total": 16, "reason": "missing_parent_hashes"}
        else:
            cand = json.loads(args.speed_summary.read_text(encoding="utf-8")).get("greedy_hashes_16", [])
            parent = json.loads(args.parent_hashes.read_text(encoding="utf-8")).get("greedy_hashes_16", [])
            matched = sum(1 for a, b in zip(cand, parent) if a and a == b) if len(cand) == 16 and len(parent) == 16 else 0
            result = {"hash_match": 1 if matched == 16 else 0, "matched": matched, "total": 16}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return

    if not args.bundle:
        raise ValueError("--bundle is required for --mode eval")
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    items: list[dict] = bundle["gpqa_splits"][args.split]
    base_url = args.url.rstrip("/")
    model = detect_model(base_url, args.model)

    rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = [pool.submit(eval_one, base_url, model, item, args.max_tokens) for item in items]
        for fut in cf.as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"{args.split} done {len(rows)}/{len(items)} correct_so_far={sum(r['correct'] for r in rows)}", flush=True)

    rows.sort(key=lambda r: r["index"])
    correct = sum(1 for r in rows if r["correct"])
    out_obj = {
        "gpqa_set": args.split,
        "gpqa_correct": correct,
        "gpqa_total": len(rows),
        "accuracy_pct": round(100.0 * correct / len(rows), 2) if rows else 0.0,
        "evaluated_jst": dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        "items": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(json.dumps({k: out_obj[k] for k in ("gpqa_set", "gpqa_correct", "gpqa_total", "accuracy_pct")}))


if __name__ == "__main__":
    main()
