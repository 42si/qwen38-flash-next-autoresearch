#!/usr/bin/env python3
"""Fixed OpenAI-compatible inference speed and prefill benchmark.

Preserves exact request payloads, streaming token timestamps, and median aggregation
from the baseline measurement:
- 16 deterministic Python code prompts, temperature=0, top_p=1, thinking disabled,
  min_tokens=400, max_tokens=400, ignore_eos=True, stop=[]
- 1 warmup request (excluded), followed by 3 repetitions at concurrencies 1, 4, 8, 16
- Public-domain US Constitution (first 16,000 chars) prefill pass with max_tokens=1
- Client-side monotonic clock timing:
  - decode_c1_tps = median((completion_tokens - 1) / (t_last_token - t_first_token))
  - total_c{4,8,16}_tps = median(total_completion_tokens / group_wall_seconds)
  - ttft_c1_s = median(t_first_token - t_request_start)
  - prefill_tps = total_prompt_tokens / (max(t_start + ttft) - min(t_start)) at c=1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import pathlib
import statistics as st
import threading
import time
import urllib.error
import urllib.request

BENCH_DIR = pathlib.Path(__file__).resolve().parent
JST = dt.timezone(dt.timedelta(hours=9))


def now_jst() -> str:
    return dt.datetime.now(JST).isoformat(timespec="milliseconds")


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


def try_flush_cache(base_url: str) -> None:
    for ep in ("/flush_cache", "/reset_prefix_cache"):
        try:
            req = urllib.request.Request(f"{base_url}{ep}", data=b"", method="POST")
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass


def send_stream(base_url: str, i: int, body: dict, gate: threading.Barrier) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    gate.wait(timeout=120)
    t0 = time.monotonic()
    first = last = None
    content: list[str] = []
    reason: list[str] = []
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
                    visible = delta.get("content") or ""
                    hidden = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if visible or hidden:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        content.append(visible)
                        reason.append(hidden)
                    finish = choice.get("finish_reason") or finish
    except urllib.error.HTTPError as ex:
        raise RuntimeError(f"HTTP {ex.code} " + ex.read().decode("utf-8", errors="replace")[:400]) from ex

    end = time.monotonic()
    msg = "".join(content)
    if first is None or not usage or not finish:
        raise ValueError(f"Incomplete stream: usage={usage} finish={finish}")
    gt = int(usage["completion_tokens"])
    pt = int(usage["prompt_tokens"])
    return {
        "i": i,
        "start": t0,
        "end": end,
        "ttft": first - t0,
        "active": last - first if last is not None else 0.0,
        "pt": pt,
        "gt": gt,
        "decode": (gt - 1) / (last - first) if (last is not None and last > first and gt > 1) else None,
        "finish": finish,
        "sha256": hashlib.sha256(msg.encode("utf-8")).hexdigest(),
        "reason_chars": len("".join(reason)),
        "details": usage.get("prompt_tokens_details"),
    }


def run_group(base_url: str, model: str, tag: str, prompts: list[str], settings: dict, raw_out: pathlib.Path | None) -> dict:
    try_flush_cache(base_url)
    n = len(prompts)
    barrier = threading.Barrier(n)
    started = now_jst()
    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(
                send_stream,
                base_url,
                i,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": p}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **settings,
                },
                barrier,
            )
            for i, p in enumerate(prompts)
        ]
        rows = [f.result() for f in futures]
    wall = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    rec = {
        "tag": tag,
        "start": started,
        "end": now_jst(),
        "settings": settings,
        "rows": rows,
        "wall": wall,
    }
    if raw_out is not None:
        with raw_out.open("a", encoding="utf-8") as w:
            w.write(json.dumps(rec) + "\n")
    return rec


def summarize_speed(groups: list[dict], greedy_hashes: list[str]) -> dict:
    by_key: dict[str, list[dict]] = {}
    for g in groups:
        if g["tag"] == "warmup":
            continue
        rows = g["rows"]
        n = len(rows)
        inp = sum(r["pt"] for r in rows)
        gen = sum(r["gt"] for r in rows)
        item: dict[str, float] = {
            "prompt": float(inp),
            "output": float(gen),
            "ttft": st.median(r["ttft"] for r in rows),
            "e2e": gen / g["wall"],
            "wall": g["wall"],
        }
        if all(r.get("decode") is not None for r in rows):
            item["sum_decode"] = sum(r["decode"] for r in rows)
        if g["tag"].startswith("prefill"):
            span = max(r["start"] + r["ttft"] for r in rows) - min(r["start"] for r in rows)
            if span > 0:
                item["effective_pp"] = inp / span
        prefix = g["tag"].split("-")[0]
        by_key.setdefault(f"{prefix}-c{n}", []).append(item)

    med: dict[str, dict] = {}
    for k, qs in by_key.items():
        common_fields = [p for p in qs[0] if all(p in q for q in qs)]
        med[k] = {"repeats": len(qs), **{p: st.median(q[p] for q in qs) for p in common_fields}}

    c1_runs = [q["sum_decode"] for q in by_key.get("code-c1", []) if "sum_decode" in q]
    return {
        "decode_c1_tps": med["code-c1"]["sum_decode"],
        "decode_c1_runs": c1_runs,
        "e2e_c1_tps": med["code-c1"]["e2e"],
        "ttft_c1_s": med["code-c1"]["ttft"],
        "total_c4_tps": med["code-c4"]["e2e"] if "code-c4" in med else None,
        "total_c8_tps": med["code-c8"]["e2e"] if "code-c8" in med else None,
        "total_c16_tps": med["code-c16"]["e2e"] if "code-c16" in med else None,
        "prefill_tps": med["prefill-c1"]["effective_pp"] if "prefill-c1" in med else None,
        "greedy_hashes_16": greedy_hashes,
        "groups_median": med,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:29771", help="OpenAI-compatible server base URL")
    ap.add_argument("--model", default=None, help="Served model ID (auto-detected via /v1/models if omitted)")
    ap.add_argument("--bundle", type=pathlib.Path, required=True, help="Path to private eval-bundle.json")
    ap.add_argument("--constitution", type=pathlib.Path, default=BENCH_DIR / "constitution.txt")
    ap.add_argument("--raw-out", type=pathlib.Path, default=None, help="Optional JSONL path for raw group timings")
    ap.add_argument("--summary-out", type=pathlib.Path, required=True, help="Output JSON summary path")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrencies", default="1,4,8,16")
    args = ap.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    code_prompts: list[str] = bundle["code"]
    if len(code_prompts) < 16:
        raise ValueError("Bundle must provide 16 code prompts")
    constitution = args.constitution.read_text(encoding="utf-8")
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]

    base_url = args.url.rstrip("/")
    model = detect_model(base_url, args.model)

    code_settings = {
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 400,
        "min_tokens": 400,
        "ignore_eos": True,
        "stop": [],
    }

    groups: list[dict] = []
    groups.append(run_group(base_url, model, "warmup", ["Count from one to ten."], {"temperature": 0, "max_tokens": 32}, args.raw_out))

    greedy_hashes: list[str | None] = [None] * 16
    c16_ok = True
    for rep in range(1, args.repeats + 1):
        for n in concs:
            if n == 16 and not c16_ok:
                continue
            try:
                rec = run_group(base_url, model, f"code-r{rep}-c{n}", code_prompts[:n], code_settings, args.raw_out)
                groups.append(rec)
                if rep == 1:
                    for row in rec["rows"]:
                        if 0 <= row["i"] < 16 and greedy_hashes[row["i"]] is None:
                            greedy_hashes[row["i"]] = row["sha256"]
            except Exception as ex:
                if n == 16:
                    c16_ok = False
                    print(f"WARN: c16 not accommodated ({ex}); continuing with mandatory c1/c4/c8", flush=True)
                else:
                    raise

    # Ensure all 16 greedy output hashes are populated even if c16 was skipped
    missing = [idx for idx, h in enumerate(greedy_hashes) if h is None]
    if missing:
        batch_size = 8 if 8 in concs else (4 if 4 in concs else 1)
        for start_i in range(0, len(missing), batch_size):
            chunk_idx = missing[start_i : start_i + batch_size]
            rec = run_group(
                base_url,
                model,
                f"hashfill-{start_i}",
                [code_prompts[idx] for idx in chunk_idx],
                code_settings,
                args.raw_out,
            )
            for idx, row in zip(chunk_idx, sorted(rec["rows"], key=lambda r: r["i"])):
                greedy_hashes[idx] = row["sha256"]

    prefill_concs = [n for n in concs if n != 16 or c16_ok]
    for n in prefill_concs:
        prompts = [f"Document {i + 1}. Summarize the following text in one sentence.\n" + constitution[:16000] for i in range(n)]
        try:
            rec = run_group(base_url, model, f"prefill-c{n}", prompts, {"temperature": 0, "max_tokens": 1}, args.raw_out)
            groups.append(rec)
        except Exception as ex:
            if n != 1:
                print(f"WARN: prefill-c{n} skipped ({ex})", flush=True)
            else:
                raise

    summary = summarize_speed(groups, [h or "" for h in greedy_hashes])
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("decode_c1_tps", "total_c4_tps", "total_c8_tps", "total_c16_tps", "ttft_c1_s", "prefill_tps")}))


if __name__ == "__main__":
    main()
