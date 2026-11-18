#!/usr/bin/env python3
"""Render karpathy/autoresearch-style stepwise progress chart (<site>/progress.png).

Strips all ancillary PNG text chunks (tEXt/zTXt/iTXt/tIME) after rendering so
zero local paths, usernames, or timestamps enter the committed binary.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import pathlib
import shutil
import struct
import subprocess
import sys


def _ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("QFN_PLOT_REEXEC") == "1":
        raise RuntimeError("matplotlib unavailable even after re-exec")
    env = os.environ.copy()
    env["QFN_PLOT_REEXEC"] = "1"
    priv_py = pathlib.Path(__file__).resolve().parents[2] / "private" / ".plot-venv" / "bin" / "python"
    if priv_py.exists():
        os.execve(str(priv_py), [str(priv_py), *sys.argv], env)
    uv_bin = shutil.which("uv")
    if uv_bin:
        os.execve(uv_bin, [uv_bin, "run", "--with", "matplotlib", "python3", *sys.argv], env)
    raise RuntimeError("matplotlib is not installed and uv is unavailable")

SITE_HARDWARE = {
    "spark": ("DGX Spark (GB10) x1", "DGX Spark (GB10 x1)"),
    "colab": ("RTX PRO 6000 x1", "Colab RTX PRO 6000 (G4 x1)"),
}


def strip_png_metadata(png_bytes: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    if not png_bytes.startswith(sig):
        raise ValueError("Not a valid PNG byte stream")
    out = bytearray(sig)
    pos = 8
    drop_types = {b"tEXt", b"zTXt", b"iTXt", b"tIME"}
    while pos < len(png_bytes):
        length = struct.unpack(">I", png_bytes[pos : pos + 4])[0]
        ctype = png_bytes[pos + 4 : pos + 8]
        chunk_end = pos + 12 + length
        if ctype not in drop_types:
            out.extend(png_bytes[pos:chunk_end])
        pos = chunk_end
        if ctype == b"IEND":
            break
    return bytes(out)


def load_rows(tsv_path: pathlib.Path) -> list[dict[str, str]]:
    if not tsv_path.exists():
        return []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_verified_records(wr_path: pathlib.Path, hw_key: str) -> list[tuple[str, float]]:
    if not wr_path.exists():
        return []
    out: list[tuple[str, float]] = []
    with wr_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("hardware") == hw_key and row.get("scope") == "single" and row.get("status") == "verified":
                try:
                    out.append((row["record_id"], float(row["value"])))
                except (KeyError, ValueError):
                    pass
    return sorted(out, key=lambda x: x[1])


def render_matplotlib(site: str, rows: list[dict[str, str]], records: list[tuple[str, float]]) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, hw_title = SITE_HARDWARE[site]
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfcfe")

    keep_x: list[int] = []
    keep_y: list[float] = []
    keep_labels: list[str] = []
    disc_x: list[int] = []
    disc_y: list[float] = []
    crash_x: list[int] = []

    for r in rows:
        try:
            t = int(r["trial"])
        except (KeyError, ValueError):
            continue
        status = r.get("status", "")
        val_str = r.get("decode_c1_tps", "")
        val = float(val_str) if val_str not in ("", None) else 0.0
        if status == "keep" and val > 0:
            keep_x.append(t)
            keep_y.append(val)
            short_sha = (r.get("commit") or "")[:7]
            keep_labels.append(f"#{t} {val:.1f}" + (f" ({short_sha})" if short_sha else ""))
        elif status == "discard":
            disc_x.append(t)
            disc_y.append(val)
        elif status == "crash":
            crash_x.append(t)

    max_trial = max([0] + keep_x + disc_x + crash_x)
    x_right = max(max_trial + 1, 5)

    palette = ["#2b6cb0", "#c53030", "#6b46c1"]
    for idx, (rec_id, rec_val) in enumerate(records):
        color = palette[idx % len(palette)]
        ax.axhline(rec_val, linestyle="--", linewidth=1.3, color=color, alpha=0.78, label=f"{rec_id}: {rec_val:.2f} tok/s")

    if disc_x:
        ax.scatter(disc_x, disc_y, color="#94a3b8", alpha=0.55, s=34, zorder=2, label="discard")
    if crash_x:
        ax.scatter(crash_x, [0.0] * len(crash_x), marker="x", color="#dc2626", s=46, linewidths=1.6, zorder=3, label="crash")

    if keep_x:
        running_best: list[float] = []
        cur = 0.0
        for y in keep_y:
            cur = max(cur, y)
            running_best.append(cur)
        step_x = keep_x + [max_trial + 0.6]
        step_y = running_best + [running_best[-1]]
        ax.step(step_x, step_y, where="post", color="#0f766e", linewidth=2.4, zorder=4, label="keep (frontier)")
        ax.scatter(keep_x, keep_y, color="#0f766e", s=54, zorder=5)
        for x, y, lbl in zip(keep_x, keep_y, keep_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 7), fontsize=8.5, color="#134e4a", fontweight="bold")
    else:
        ax.text(
            0.5,
            0.5,
            "No completed trials recorded yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
        )

    all_y = keep_y + disc_y + [v for _, v in records]
    y_top = max(all_y) * 1.15 if all_y else 140.0
    ax.set_xlim(-0.4, x_right)
    ax.set_ylim(-2.0, max(y_top, 50.0))
    ax.set_xlabel("Trial", fontsize=10.5)
    ax.set_ylabel("Single-stream decode throughput — decode_c1_tps (tok/s)", fontsize=10.5)
    ax.set_title(f"Qwen3.8-Flash-Next Autoresearch Progress — {hw_title}", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(loc="lower right" if keep_y and max(keep_y) > 80 else "upper left", frameon=True, facecolor="#ffffff", framealpha=0.92, fontsize=8.8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={})
    plt.close(fig)
    return strip_png_metadata(buf.getvalue())


def main() -> None:
    _ensure_matplotlib()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", choices=["spark", "colab"], required=True)
    ap.add_argument("--tsv", type=pathlib.Path, required=True)
    ap.add_argument("--world-records", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    hw_key, _ = SITE_HARDWARE[args.site]
    rows = load_rows(args.tsv)
    records = load_verified_records(args.world_records, hw_key)
    png_bytes = render_matplotlib(args.site, rows, records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(png_bytes)
    print(f"Wrote {args.out} ({len(png_bytes)} bytes, {len(rows)} rows)")


if __name__ == "__main__":
    main()
