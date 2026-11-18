#!/usr/bin/env bash
# Fixed 1-trial runner for karpathy/autoresearch loop (<spark|colab>).
# Do not edit during trials; bench/FROZEN.sha256 is verified on every invocation.
set -euo pipefail

SITE="${1:?Usage: bench/run_trial.sh <spark|colab> [description_override]}"
DESC_OVERRIDE="${2:-}"
if [[ "$SITE" != "spark" && "$SITE" != "colab" ]]; then
  echo "ERROR: site must be 'spark' or 'colab'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$LOOP_DIR/.." && pwd)"
MAIN_REPO_DIR="${QFN_MAIN_REPO_DIR:-$ROOT_DIR/repo}"
PRIVATE_DIR="${QFN_PRIVATE_DIR:-$ROOT_DIR/private}"

if [[ -f "$PRIVATE_DIR/host.env" ]]; then
  # shellcheck disable=SC1090
  source "$PRIVATE_DIR/host.env"
fi

PORT="${QFN_PORT:-29771}"
AGENT_LABEL="${QFN_AGENT_MODEL:-manual}"

# 1. Verify frozen bench/ manifest
(cd "$MAIN_REPO_DIR" && shasum -a 256 -c bench/FROZEN.sha256 >/dev/null)
(cd "$LOOP_DIR" && shasum -a 256 -c bench/FROZEN.sha256 >/dev/null)

# Sync any unstaged research-log additions from loop worktree into canonical main repo
if ! git -C "$LOOP_DIR" diff --quiet -- records/research-log.md 2>/dev/null; then
  git -C "$LOOP_DIR" diff --unified=0 -- records/research-log.md \
    | awk '/^\+[^+]/ {print substr($0, 2)}' >> "$MAIN_REPO_DIR/records/research-log.md"
  git -C "$LOOP_DIR" checkout -- records/research-log.md
fi

TSV_PATH="$MAIN_REPO_DIR/$SITE/results.tsv"
PNG_PATH="$MAIN_REPO_DIR/$SITE/progress.png"
WR_PATH="$MAIN_REPO_DIR/records/world-records.tsv"
BUNDLE_PATH="$PRIVATE_DIR/eval-bundle.json"
PARENT_HASHES_PATH="$PRIVATE_DIR/parent-hashes-$SITE.json"

if [[ ! -f "$BUNDLE_PATH" ]]; then
  python3 "$MAIN_REPO_DIR/bench/prepare_inputs.py" --out "$BUNDLE_PATH"
fi

# Read parent keep state from canonical results.tsv
STATE_JSON="$(python3 - "$TSV_PATH" <<'PY'
import csv, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(p.open(encoding="utf-8"), delimiter="\t")) if p.exists() else []
next_trial = 0 if not rows else max(int(r["trial"]) for r in rows) + 1
keeps = [r for r in rows if r["status"] == "keep" and r.get("decode_c1_tps")]
parent = keeps[-1] if keeps else {}
best_decode = max((float(r["decode_c1_tps"]) for r in keeps), default=0.0)
a0 = next((r for r in keeps if r.get("gpqa_set") == "G198" and r.get("gpqa_correct")), {})
g40_parent = next((r for r in reversed(keeps) if r.get("gpqa_set") in ("G40", "G198") and r.get("gpqa_correct")), parent)
print(json.dumps({
    "next_trial": next_trial,
    "parent_commit": parent.get("commit", ""),
    "parent_decode": float(parent["decode_c1_tps"]) if parent.get("decode_c1_tps") else 0.0,
    "parent_c4": float(parent["total_c4_tps"]) if parent.get("total_c4_tps") else 0.0,
    "parent_c8": float(parent["total_c8_tps"]) if parent.get("total_c8_tps") else 0.0,
    "parent_ttft": float(parent["ttft_c1_s"]) if parent.get("ttft_c1_s") else 0.0,
    "parent_gpqa_set": parent.get("gpqa_set", ""),
    "parent_gpqa_correct": parent.get("gpqa_correct", ""),
    "parent_gpqa_total": parent.get("gpqa_total", ""),
    "g40_ref_correct": int(g40_ref["gpqa_correct"]) if (g40_ref := g40_parent).get("gpqa_correct") else None,
    "g40_ref_set": g40_parent.get("gpqa_set", ""),
    "a0_g198_correct": int(a0["gpqa_correct"]) if a0.get("gpqa_correct") else None,
    "best_decode": best_decode,
}))
PY
)"

TRIAL_NUM="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["next_trial"])' "$STATE_JSON")"
PARENT_COMMIT="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["parent_commit"])' "$STATE_JSON")"
CAND_COMMIT="$(git -C "$LOOP_DIR" rev-parse HEAD)"
STARTED_JST="$(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S')"
COMMIT_SUBJECT="$(git -C "$LOOP_DIR" log -1 --format=%s)"
DESCRIPTION="${DESC_OVERRIDE:-$COMMIT_SUBJECT}"

# Extract candidate metadata
META_JSON="$(bash "$LOOP_DIR/$SITE/candidate.sh" meta)"
ENGINE="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("engine",""))' "$META_JSON")"
ENGINE_REV="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("engine_rev",""))' "$META_JSON")"
QUANT="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("quant",""))' "$META_JSON")"
WEIGHTS_REV="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("weights_rev",""))' "$META_JSON")"

finalize_and_push() {
  local row_json="$1"
  local status="$2"
  python3 "$MAIN_REPO_DIR/bench/record.py" --tsv "$TSV_PATH" --row-json "$row_json"
  python3 "$MAIN_REPO_DIR/bench/plot_progress.py" --site "$SITE" --tsv "$TSV_PATH" --world-records "$WR_PATH" --out "$PNG_PATH"

  git -C "$MAIN_REPO_DIR" add "$SITE/results.tsv" "$SITE/progress.png" records/research-log.md
  bash "$MAIN_REPO_DIR/bench/scan_secrets.sh" "$MAIN_REPO_DIR"
  git -C "$MAIN_REPO_DIR" commit -m "$SITE trial $TRIAL_NUM: $status"
  git -C "$MAIN_REPO_DIR" push origin main

  if [[ "$status" == "keep" ]]; then
    bash "$MAIN_REPO_DIR/bench/scan_secrets.sh" "$LOOP_DIR"
    git -C "$LOOP_DIR" push origin "HEAD:refs/heads/autoresearch/${SITE}-20261119" || true
  else
    git -C "$LOOP_DIR" tag -f "trial/$SITE/$TRIAL_NUM" "$CAND_COMMIT"
    bash "$MAIN_REPO_DIR/bench/scan_secrets.sh" "$LOOP_DIR"
    git -C "$LOOP_DIR" push -f origin "trial/$SITE/$TRIAL_NUM" || true
    if [[ -n "$PARENT_COMMIT" ]]; then
      git -C "$LOOP_DIR" reset --hard "$PARENT_COMMIT"
    fi
  fi
}

# 2. Enforce single-file rule (<site>/candidate.sh only)
DIRTY="$(git -C "$LOOP_DIR" status --porcelain --untracked-files=no)"
if [[ -n "$DIRTY" ]]; then
  echo "ERROR: loop worktree has uncommitted tracked modifications; commit $SITE/candidate.sh first" >&2
  exit 2
fi

if [[ -n "$PARENT_COMMIT" ]]; then
  CHANGED_FILES="$(git -C "$LOOP_DIR" diff --name-only "$PARENT_COMMIT" "$CAND_COMMIT")"
  if [[ "$CHANGED_FILES" != "$SITE/candidate.sh" ]]; then
    TMP_ROW="$(mktemp)"
    python3 - "$TMP_ROW" "$TRIAL_NUM" "$CAND_COMMIT" "$STARTED_JST" "$ENGINE" "$ENGINE_REV" "$QUANT" "$WEIGHTS_REV" "$PARENT_COMMIT" "$AGENT_LABEL" "$CHANGED_FILES" <<'PY'
import json, pathlib, sys
out, trial, commit, started, eng, erev, quant, wrev, parent, agent, changed = sys.argv[1:12]
desc = "single-file rule violation: modified [" + (", ".join(changed.splitlines()) or "none") + "]"
pathlib.Path(out).write_text(json.dumps({
    "trial": trial, "commit": commit[:8], "started_jst": started,
    "decode_c1_tps": "0", "total_c4_tps": "0", "total_c8_tps": "0", "total_c16_tps": "0",
    "ttft_c1_s": "0", "prefill_tps": "0", "peak_mem_gib": "0",
    "status": "discard", "description": desc,
    "engine": eng, "engine_rev": erev, "quant": quant, "weights_rev": wrev,
    "parent": parent[:8], "agent": agent,
}))
PY
    finalize_and_push "$TMP_ROW" "discard"
    rm -f "$TMP_ROW"
    exit 0
  fi
fi

# 3. Target machine pre-flight gate & remote execution
if [[ "$SITE" == "colab" ]]; then
  CU_OK="$(usage --json --verbose --providers colab | python3 -c '
import json, sys
d = json.load(sys.stdin)
r = d["rows"][0]
ok = (not r.get("stale", True)) and float(r.get("remainingCu", 0)) >= 2020.0
print("1" if ok else "0")
')"
  if [[ "$CU_OK" != "1" ]]; then
    echo "STOP: Colab CU balance below 2020 CU safety floor or stale" >&2
    exit 4
  fi
  echo "STOP: Colab runtime execution requires interactive colab-cli authorization" >&2
  exit 3
fi

SSH_TARGET="${QFN_SSH_TARGET:?QFN_SSH_TARGET must be set in private/host.env}"
# shellcheck disable=SC2206
SSH_OPTS=(${QFN_SSH_OPTS:--o BatchMode=yes -o ClearAllForwardings=yes})
REMOTE_DIR="${QFN_REMOTE_DIR:-\$HOME/qfn-autoresearch}"

# Gate check on remote GB10 host: no trial lock, no GPU compute apps, no foreign inference servers, >=85 GiB free
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "PORT='$PORT' bash -s" <<'EOF'
set -euo pipefail
if tmux has-session -t qfn-ar-spark-trial 2>/dev/null; then
  echo "GATE_BUSY: qfn-ar-spark-trial tmux lock active" >&2
  exit 3
fi
if ss -H -ltn "sport = :$PORT or sport = :29761" | grep -q .; then
  echo "GATE_BUSY: measurement port occupied" >&2
  exit 3
fi
APPS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
if [[ -n "$APPS" ]]; then
  echo "GATE_BUSY: GPU compute apps active: $APPS" >&2
  exit 3
fi
if pgrep -u "$(id -u)" -fa 'vllm|sglang|llama-server|tensorfold-native|hibana|trtllm' | grep -v 'pgrep' >/dev/null 2>&1; then
  echo "GATE_BUSY: existing inference server process detected on host" >&2
  exit 3
fi
AVAIL_KB="$(awk '$1=="MemAvailable:"{print $2}' /proc/meminfo)"
if (( AVAIL_KB < 85 * 1024 * 1024 )); then
  echo "GATE_BUSY: insufficient MemAvailable (${AVAIL_KB} kB < 85 GiB)" >&2
  exit 3
fi
EOF

# Sync bench scripts, candidate.sh, and private bundle to remote workspace
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p $REMOTE_DIR/{bench,private,runs,models,engines}"
scp -q "${SSH_OPTS[@]}" \
  "$MAIN_REPO_DIR/bench/measure.py" \
  "$MAIN_REPO_DIR/bench/guard.py" \
  "$MAIN_REPO_DIR/bench/constitution.txt" \
  "$MAIN_REPO_DIR/bench/guard-ids.sha256" \
  "$SSH_TARGET:${REMOTE_DIR}/bench/"
scp -q "${SSH_OPTS[@]}" "$LOOP_DIR/$SITE/candidate.sh" "$SSH_TARGET:${REMOTE_DIR}/candidate.sh"
scp -q "${SSH_OPTS[@]}" "$BUNDLE_PATH" "$SSH_TARGET:${REMOTE_DIR}/private/eval-bundle.json"
if [[ -f "$PARENT_HASHES_PATH" ]]; then
  scp -q "${SSH_OPTS[@]}" "$PARENT_HASHES_PATH" "$SSH_TARGET:${REMOTE_DIR}/private/parent-hashes.json"
fi

run_remote_pass() {
  local run_tag="$1"
  local eval_split="$2"
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "RUN_TAG='$run_tag' PORT='$PORT' EVAL_SPLIT='$eval_split' WORKDIR=\"$REMOTE_DIR\" bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$WORKDIR"
RDIR="runs/$RUN_TAG"
rm -rf "$RDIR"
mkdir -p "$RDIR"

cat > "$RDIR/driver.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd "$WORKDIR"
RDIR="runs/$RUN_TAG"
export QFN_WORKDIR="$WORKDIR"
export QFN_PORT="$PORT"

cleanup() {
  if [[ -f "$RDIR/server.pid" ]]; then
    SPID="$(cat "$RDIR/server.pid" || true)"
    if [[ -n "$SPID" ]]; then
      kill -TERM "-$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null || true
      for _ in $(seq 1 45); do
        kill -0 "$SPID" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "-$SPID" 2>/dev/null || kill -KILL "$SPID" 2>/dev/null || true
    fi
    rm -f "$RDIR/server.pid"
  fi
  if [[ -f "$RDIR/mem.pid" ]]; then
    kill -TERM "$(cat "$RDIR/mem.pid")" 2>/dev/null || true
    rm -f "$RDIR/mem.pid"
  fi
}
trap cleanup EXIT

echo "prepare" > "$RDIR/phase"
if ! timeout -k 30 1200 bash candidate.sh prepare > "$RDIR/prepare.log" 2>&1; then
  echo "prepare_failed" > "$RDIR/error"
  exit 1
fi

if [[ "$EVAL_SPLIT" == "NONE" ]]; then
  # Speed + prefill measurement pass
  BASE_MEM_KB="$(awk '$1=="MemAvailable:"{print $2}' /proc/meminfo)"
  (
    PEAK_USED_KB=0
    while true; do
      CUR_AVAIL_KB="$(awk '$1=="MemAvailable:"{print $2}' /proc/meminfo)"
      TOTAL_KB="$(awk '$1=="MemTotal:"{print $2}' /proc/meminfo)"
      USED_KB=$(( TOTAL_KB - CUR_AVAIL_KB ))
      if (( USED_KB > PEAK_USED_KB )); then
        PEAK_USED_KB=$USED_KB
        awk -v kb="$PEAK_USED_KB" 'BEGIN { printf "%.2f\n", kb / 1048576.0 }' > "$RDIR/peak_mem_gib"
      fi
      sleep 2
    done
  ) &
  echo $! > "$RDIR/mem.pid"

  echo "start_speed" > "$RDIR/phase"
  setsid bash candidate.sh start speed > "$RDIR/server.log" 2>&1 < /dev/null &
  SPID=$!
  echo "$SPID" > "$RDIR/server.pid"

  READY=0
  for _ in $(seq 1 600); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$SPID" 2>/dev/null; then
      echo "server_exited_during_startup" > "$RDIR/error"
      exit 1
    fi
    sleep 2
  done
  if (( READY == 0 )); then
    echo "startup_timeout_1200s" > "$RDIR/error"
    exit 1
  fi

  echo "measure_speed" > "$RDIR/phase"
  if ! timeout -k 15 600 python3 bench/measure.py \
      --url "http://127.0.0.1:$PORT" \
      --bundle private/eval-bundle.json \
      --raw-out "$RDIR/speed.jsonl" \
      --summary-out "$RDIR/speed-summary.json" > "$RDIR/measure.log" 2>&1; then
    echo "measure_timeout_or_error" > "$RDIR/error"
    exit 1
  fi

  python3 bench/guard.py \
    --mode check-hash \
    --speed-summary "$RDIR/speed-summary.json" \
    --parent-hashes private/parent-hashes.json \
    --out "$RDIR/hash-check.json" >> "$RDIR/measure.log" 2>&1 || true

  cleanup
else
  # GPQA guard evaluation pass (G40 or G198)
  echo "start_guard" > "$RDIR/phase"
  setsid bash candidate.sh start guard > "$RDIR/server-guard.log" 2>&1 < /dev/null &
  SPID=$!
  echo "$SPID" > "$RDIR/server.pid"

  READY=0
  for _ in $(seq 1 600); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$SPID" 2>/dev/null; then
      echo "guard_server_exited_during_startup" > "$RDIR/error"
      exit 1
    fi
    sleep 2
  done
  if (( READY == 0 )); then
    echo "guard_startup_timeout_1200s" > "$RDIR/error"
    exit 1
  fi

  CAP_SEC=1200
  if [[ "$EVAL_SPLIT" == "G198" ]]; then CAP_SEC=5400; fi
  if ! timeout -k 15 "$CAP_SEC" python3 bench/guard.py \
      --mode eval \
      --url "http://127.0.0.1:$PORT" \
      --bundle private/eval-bundle.json \
      --split "$EVAL_SPLIT" \
      --out "$RDIR/guard-${EVAL_SPLIT}.json" > "$RDIR/guard.log" 2>&1; then
    echo "guard_timeout_or_error_${EVAL_SPLIT}" > "$RDIR/error"
    exit 1
  fi
  cleanup
fi
echo "ok" > "$RDIR/done"
INNER
chmod +x "$RDIR/driver.sh"

tmux new-session -d -s qfn-ar-spark-trial "WORKDIR='$WORKDIR' RUN_TAG='$RUN_TAG' PORT='$PORT' EVAL_SPLIT='$EVAL_SPLIT' bash '$WORKDIR/$RDIR/driver.sh'; echo \$? > '$WORKDIR/$RDIR/exit_code'"
while tmux has-session -t qfn-ar-spark-trial 2>/dev/null; do
  sleep 5
done
EC="$(cat "$RDIR/exit_code" 2>/dev/null || echo 1)"
exit "$EC"
REMOTE_SCRIPT
}

LOCAL_RUN_DIR="$PRIVATE_DIR/runs/$SITE-$TRIAL_NUM"
rm -rf "$LOCAL_RUN_DIR"
mkdir -p "$LOCAL_RUN_DIR"

if ! run_remote_pass "trial-$TRIAL_NUM" "NONE"; then
  ERR_REASON="$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cat $REMOTE_DIR/runs/trial-$TRIAL_NUM/error 2>/dev/null || echo runtime_crash")"
  TMP_ROW="$(mktemp)"
  python3 - "$TMP_ROW" "$TRIAL_NUM" "$CAND_COMMIT" "$STARTED_JST" "$DESCRIPTION ($ERR_REASON)" "$ENGINE" "$ENGINE_REV" "$QUANT" "$WEIGHTS_REV" "$PARENT_COMMIT" "$AGENT_LABEL" <<'PY'
import json, pathlib, sys
out, trial, commit, started, desc, eng, erev, quant, wrev, parent, agent = sys.argv[1:12]
pathlib.Path(out).write_text(json.dumps({
    "trial": trial, "commit": commit[:8], "started_jst": started,
    "decode_c1_tps": "0", "total_c4_tps": "0", "total_c8_tps": "0", "total_c16_tps": "0",
    "ttft_c1_s": "0", "prefill_tps": "0", "peak_mem_gib": "0",
    "status": "crash", "description": desc,
    "engine": eng, "engine_rev": erev, "quant": quant, "weights_rev": wrev,
    "parent": parent[:8], "agent": agent,
}))
PY
  finalize_and_push "$TMP_ROW" "crash"
  rm -f "$TMP_ROW"
  exit 0
fi

scp -q "${SSH_OPTS[@]}" \
  "$SSH_TARGET:${REMOTE_DIR}/runs/trial-$TRIAL_NUM/speed-summary.json" \
  "$SSH_TARGET:${REMOTE_DIR}/runs/trial-$TRIAL_NUM/hash-check.json" \
  "$SSH_TARGET:${REMOTE_DIR}/runs/trial-$TRIAL_NUM/peak_mem_gib" \
  "$LOCAL_RUN_DIR/"

# Evaluate speed thresholds before deciding whether G40/G198 & cold-restart reproduction are needed
EVAL_DECISION="$(python3 - "$STATE_JSON" "$LOCAL_RUN_DIR/speed-summary.json" "$LOCAL_RUN_DIR/hash-check.json" <<'PY'
import json, pathlib, sys
st = json.loads(sys.argv[1])
sp = json.loads(pathlib.Path(sys.argv[2]).read_text())
hc = json.loads(pathlib.Path(sys.argv[3]).read_text())

dec = float(sp["decode_c1_tps"])
c4 = sp.get("total_c4_tps")
c8 = sp.get("total_c8_tps")
ttft = float(sp["ttft_c1_s"])
hash_match = int(hc.get("hash_match", 0))

p_dec = st["parent_decode"]
if p_dec <= 0:
    print(json.dumps({"speed_ok": True, "gain_pct": 0.0, "need_g40": hash_match == 0, "need_g198": False}))
    sys.exit(0)

gain_pct = 100.0 * (dec - p_dec) / p_dec
c4_ok = (c4 is not None) and (not st["parent_c4"] or c4 >= 0.97 * st["parent_c4"])
c8_ok = (c8 is not None) and (not st["parent_c8"] or c8 >= 0.97 * st["parent_c8"])
ttft_ok = (not st["parent_ttft"]) or (ttft <= 1.10 * st["parent_ttft"])
speed_ok = (gain_pct >= 3.0) and c4_ok and c8_ok and ttft_ok

print(json.dumps({
    "speed_ok": speed_ok,
    "gain_pct": gain_pct,
    "c4_ok": c4_ok,
    "c8_ok": c8_ok,
    "ttft_ok": ttft_ok,
    "hash_match": hash_match,
    "need_g40": speed_ok and (hash_match == 0),
    "need_g198": speed_ok and (hash_match == 0) and (dec > st["best_decode"]),
}))
PY
)"

SPEED_OK="$(python3 -c 'import json,sys;print("1" if json.loads(sys.argv[1])["speed_ok"] else "0")' "$EVAL_DECISION")"
NEED_G40="$(python3 -c 'import json,sys;print("1" if json.loads(sys.argv[1])["need_g40"] else "0")' "$EVAL_DECISION")"
NEED_G198="$(python3 -c 'import json,sys;print("1" if json.loads(sys.argv[1])["need_g198"] else "0")' "$EVAL_DECISION")"
HASH_MATCH="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("hash_match",0))' "$EVAL_DECISION")"

GUARD_OK=1
GPQA_SET=""
GPQA_CORRECT=""
GPQA_TOTAL=""

if [[ "$HASH_MATCH" == "1" ]]; then
  GPQA_SET="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["parent_gpqa_set"])' "$STATE_JSON")"
  GPQA_CORRECT="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["parent_gpqa_correct"])' "$STATE_JSON")"
  GPQA_TOTAL="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["parent_gpqa_total"])' "$STATE_JSON")"
elif [[ "$NEED_G40" == "1" ]]; then
  if run_remote_pass "trial-${TRIAL_NUM}-g40" "G40"; then
    scp -q "${SSH_OPTS[@]}" "$SSH_TARGET:${REMOTE_DIR}/runs/trial-${TRIAL_NUM}-g40/guard-G40.json" "$LOCAL_RUN_DIR/"
    GPQA_SET="G40"
    GPQA_CORRECT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["gpqa_correct"])' "$LOCAL_RUN_DIR/guard-G40.json")"
    GPQA_TOTAL="40"
    G40_PASS="$(python3 - "$STATE_JSON" "$GPQA_CORRECT" <<'PY'
import json, sys
st = json.loads(sys.argv[1])
cur = int(sys.argv[2])
ref = st.get("g40_ref_correct")
if ref is None or st.get("g40_ref_set") != "G40":
    print("1" if cur >= 34 else "0")
else:
    print("1" if cur >= ref - 2 else "0")
PY
)"
    if [[ "$G40_PASS" != "1" ]]; then GUARD_OK=0; fi
  else
    GUARD_OK=0
  fi

  if [[ "$GUARD_OK" == "1" && "$NEED_G198" == "1" ]]; then
    if run_remote_pass "trial-${TRIAL_NUM}-g198" "G198"; then
      scp -q "${SSH_OPTS[@]}" "$SSH_TARGET:${REMOTE_DIR}/runs/trial-${TRIAL_NUM}-g198/guard-G198.json" "$LOCAL_RUN_DIR/"
      GPQA_SET="G198"
      GPQA_CORRECT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["gpqa_correct"])' "$LOCAL_RUN_DIR/guard-G198.json")"
      GPQA_TOTAL="198"
      G198_PASS="$(python3 - "$STATE_JSON" "$GPQA_CORRECT" <<'PY'
import json, sys
st = json.loads(sys.argv[1])
cur = int(sys.argv[2])
ref = st.get("a0_g198_correct")
if ref is None:
    print("1" if cur >= 175 else "0")
else:
    print("1" if cur >= ref - 2 else "0")
PY
)"
      if [[ "$G198_PASS" != "1" ]]; then GUARD_OK=0; fi
    else
      GUARD_OK=0
    fi
  fi
fi

# Independent 2nd cold-start reproduction check before accepting keep
FINAL_STATUS="discard"
if [[ "$SPEED_OK" == "1" && "$GUARD_OK" == "1" ]]; then
  if [[ -z "$PARENT_COMMIT" ]]; then
    FINAL_STATUS="keep"
  elif run_remote_pass "trial-${TRIAL_NUM}-repro" "NONE"; then
    scp -q "${SSH_OPTS[@]}" "$SSH_TARGET:${REMOTE_DIR}/runs/trial-${TRIAL_NUM}-repro/speed-summary.json" "$LOCAL_RUN_DIR/repro-summary.json"
    REPRO_OK="$(python3 - "$STATE_JSON" "$LOCAL_RUN_DIR/repro-summary.json" <<'PY'
import json, pathlib, sys
st = json.loads(sys.argv[1])
sp = json.loads(pathlib.Path(sys.argv[2]).read_text())
dec = float(sp["decode_c1_tps"])
c4 = sp.get("total_c4_tps")
c8 = sp.get("total_c8_tps")
ttft = float(sp["ttft_c1_s"])
gain_pct = 100.0 * (dec - st["parent_decode"]) / st["parent_decode"]
c4_ok = (c4 is not None) and (not st["parent_c4"] or c4 >= 0.97 * st["parent_c4"])
c8_ok = (c8 is not None) and (not st["parent_c8"] or c8 >= 0.97 * st["parent_c8"])
ttft_ok = (not st["parent_ttft"]) or (ttft <= 1.10 * st["parent_ttft"])
print("1" if (gain_pct >= 3.0 and c4_ok and c8_ok and ttft_ok) else "0")
PY
)"
    if [[ "$REPRO_OK" == "1" ]]; then
      FINAL_STATUS="keep"
    else
      DESCRIPTION="$DESCRIPTION (repro check below threshold)"
    fi
  else
    DESCRIPTION="$DESCRIPTION (repro cold-start failed)"
  fi
fi

if [[ "$FINAL_STATUS" == "keep" ]]; then
  cp "$LOCAL_RUN_DIR/speed-summary.json" "$PARENT_HASHES_PATH"
fi

TMP_ROW="$(mktemp)"
python3 - "$TMP_ROW" "$TRIAL_NUM" "$CAND_COMMIT" "$STARTED_JST" "$LOCAL_RUN_DIR" "$STATE_JSON" "$FINAL_STATUS" "$DESCRIPTION" "$ENGINE" "$ENGINE_REV" "$QUANT" "$WEIGHTS_REV" "$AGENT_LABEL" "$HASH_MATCH" "$GPQA_SET" "$GPQA_CORRECT" "$GPQA_TOTAL" <<'PY'
import json, pathlib, statistics as st_mod, sys

(
    out_path, trial, commit, started, run_dir_s, state_s, status, desc,
    eng, erev, quant, wrev, agent, hash_match, gpqa_set, gpqa_correct, gpqa_total
) = sys.argv[1:18]
run_dir = pathlib.Path(run_dir_s)
state = json.loads(state_s)
sp1 = json.loads((run_dir / "speed-summary.json").read_text())
repro_p = run_dir / "repro-summary.json"
if repro_p.exists():
    sp2 = json.loads(repro_p.read_text())
    dec = min(float(sp1["decode_c1_tps"]), float(sp2["decode_c1_tps"]))
    c4 = min(float(sp1["total_c4_tps"]), float(sp2["total_c4_tps"])) if sp1.get("total_c4_tps") and sp2.get("total_c4_tps") else sp1.get("total_c4_tps")
    c8 = min(float(sp1["total_c8_tps"]), float(sp2["total_c8_tps"])) if sp1.get("total_c8_tps") and sp2.get("total_c8_tps") else sp1.get("total_c8_tps")
    c16 = min(float(sp1["total_c16_tps"]), float(sp2["total_c16_tps"])) if sp1.get("total_c16_tps") and sp2.get("total_c16_tps") else sp1.get("total_c16_tps")
    ttft = max(float(sp1["ttft_c1_s"]), float(sp2["ttft_c1_s"]))
    pp = min(float(sp1["prefill_tps"]), float(sp2["prefill_tps"])) if sp1.get("prefill_tps") and sp2.get("prefill_tps") else sp1.get("prefill_tps")
else:
    dec = float(sp1["decode_c1_tps"])
    c4 = sp1.get("total_c4_tps")
    c8 = sp1.get("total_c8_tps")
    c16 = sp1.get("total_c16_tps")
    ttft = float(sp1["ttft_c1_s"])
    pp = sp1.get("prefill_tps")

peak_mem = (run_dir / "peak_mem_gib").read_text().strip() if (run_dir / "peak_mem_gib").exists() else ""
p_dec = state["parent_decode"]
gain = f"{100.0 * (dec - p_dec) / p_dec:+.2f}" if p_dec > 0 else ""

row = {
    "trial": trial,
    "commit": commit[:8],
    "started_jst": started,
    "decode_c1_tps": f"{dec:.2f}",
    "total_c4_tps": f"{c4:.2f}" if c4 is not None else "",
    "total_c8_tps": f"{c8:.2f}" if c8 is not None else "",
    "total_c16_tps": f"{c16:.2f}" if c16 is not None else "",
    "ttft_c1_s": f"{ttft:.3f}",
    "prefill_tps": f"{pp:.2f}" if pp is not None else "",
    "gpqa_set": gpqa_set,
    "gpqa_correct": gpqa_correct,
    "gpqa_total": gpqa_total,
    "hash_match": hash_match,
    "peak_mem_gib": peak_mem,
    "status": status,
    "description": desc,
    "engine": eng,
    "engine_rev": erev,
    "quant": quant,
    "weights_rev": wrev,
    "parent": state["parent_commit"][:8] if state["parent_commit"] else "",
    "gain_vs_parent_pct": gain,
    "wr_ref": "",
    "vs_wr_pct": "",
    "agent": agent,
    "cu_spent": "",
}
pathlib.Path(out_path).write_text(json.dumps(row, indent=2), encoding="utf-8")
PY

finalize_and_push "$TMP_ROW" "$FINAL_STATUS"
rm -f "$TMP_ROW"
