#!/usr/bin/env bash
# Continuous karpathy/autoresearch loop controller (<spark|colab>).
# Checks quota before each 8-trial round, selects an eligible implementation model
# explicitly (never Fable, never GPT-6.x-Sol, never agent-run auto), and loops continuously.
set -euo pipefail

SITE="${1:?Usage: bash bench/loop.sh <spark|colab>}"
if [[ "$SITE" != "spark" && "$SITE" != "colab" ]]; then
  echo "ERROR: site must be 'spark' or 'colab'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$LOOP_DIR/.." && pwd)"
MAIN_REPO_DIR="${QFN_MAIN_REPO_DIR:-$ROOT_DIR/repo}"
PRIVATE_DIR="${QFN_PRIVATE_DIR:-$ROOT_DIR/private}"
AGENT_RUN_BIN="${QFN_AGENT_RUN:-$HOME/agent-hub/scripts/agent-run}"

mkdir -p "$PRIVATE_DIR"
if [[ -f "$PRIVATE_DIR/host.env" ]]; then
  # shellcheck disable=SC1090
  source "$PRIVATE_DIR/host.env"
fi

log() {
  printf '[%s JST] [loop:%s] %s\n' "$(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S')" "$SITE" "$*"
}

ROUND=1
while true; do
  if [[ -f "$LOOP_DIR/STOP" || -f "$MAIN_REPO_DIR/STOP" || -f "$PRIVATE_DIR/STOP-$SITE" ]]; then
    log "STOP marker detected; halting loop cleanly."
    exit 0
  fi

  # Verify bench/ integrity before starting round
  if ! (cd "$MAIN_REPO_DIR" && shasum -a 256 -c bench/FROZEN.sha256 >/dev/null 2>&1); then
    log "STOP: bench/FROZEN.sha256 verification failed in main repo."
    exit 2
  fi
  if ! (cd "$LOOP_DIR" && shasum -a 256 -c bench/FROZEN.sha256 >/dev/null 2>&1); then
    log "STOP: bench/FROZEN.sha256 verification failed in loop worktree."
    exit 2
  fi

  # Pre-round host reachability & exclusive-GPU gate on spark (avoids burning LLM quota if host is offline or busy)
  if [[ "$SITE" == "spark" && -n "${QFN_SSH_TARGET:-}" ]]; then
    # shellcheck disable=SC2206
    SSH_OPTS=(${QFN_SSH_OPTS:--o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=15})
    PORT_CHECK="${QFN_PORT:-29771}"
    HOST_GATE="$(ssh "${SSH_OPTS[@]}" "$QFN_SSH_TARGET" "PORT='$PORT_CHECK' bash -s" <<'EOF' 2>/dev/null || echo "UNREACHABLE"
set -euo pipefail
if tmux has-session -t qfn-ar-spark-trial 2>/dev/null; then echo "BUSY:trial_lock"; exit 0; fi
if ss -H -ltn "sport = :$PORT or sport = :29761" | grep -q .; then echo "BUSY:port_occupied"; exit 0; fi
APPS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
if [[ -n "$APPS" ]]; then echo "BUSY:gpu_apps($APPS)"; exit 0; fi
if pgrep -u "$(id -u)" -fa 'vllm|sglang|llama-server|tensorfold-native|hibana|trtllm' | grep -v 'pgrep' >/dev/null 2>&1; then
  echo "BUSY:existing_inference_server"
  exit 0
fi
echo "READY"
EOF
)"
    if [[ "$HOST_GATE" != "READY" ]]; then
      log "Target host gate is '$HOST_GATE' (not stopping or modifying host services); sleeping 120s without spending agent quota..."
      sleep 120
      continue
    fi
  fi

  # Colab CU safety gate before round
  if [[ "$SITE" == "colab" ]]; then
    COLAB_JSON="$(usage --json --verbose --providers colab 2>/dev/null || echo '{}')"
    CU_STATUS="$(python3 - "$COLAB_JSON" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
    r = d["rows"][0]
    rem = float(r.get("remainingCu", 0.0))
    stale = bool(r.get("stale", True))
    assigns = int(r.get("assignmentsCount", 0))
    if stale:
        print("STALE")
    elif rem < 2020.0:
        print(f"FLOOR:{rem:.2f}")
    elif assigns > 1:
        print(f"ASSIGNS:{assigns}")
    else:
        print(f"OK:{rem:.2f}")
except Exception:
    print("ERR")
PY
)"
    if [[ "$CU_STATUS" != OK:* ]]; then
      log "STOP: Colab CU check failed ($CU_STATUS)."
      exit 4
    fi
  fi

  # Quota check via usage-collect JSON (fresh <= 60 min, >= 30% remaining required for normal implementation)
  USAGE_JSON="$("$HOME/.local/bin/usage-collect" --json --providers claude,codex,agy --deadline 25 --max-cache-age 120 2>/dev/null || echo '{}')"
  SELECTION="$(python3 - "$USAGE_JSON" <<'PY'
import json, sys

raw = sys.argv[1]
try:
    d = json.loads(raw)
    rows = d.get("rows", [])
except Exception:
    print("WAIT\tusage_unavailable\t0\t0\t0")
    sys.exit(0)

def get_pct(provider: str, match_substr: str) -> float | None:
    vals = []
    for r in rows:
        if r.get("provider", "").lower() != provider.lower():
            continue
        if r.get("status") != "ok" or r.get("stale", True):
            continue
        win = (r.get("window") or "") + " " + (r.get("windowId") or "")
        if match_substr.lower() in win.lower():
            p = r.get("remainingPercent")
            if isinstance(p, (int, float)):
                vals.append(float(p))
    return min(vals) if vals else None

gem_5h = get_pct("agy", "5h")
gem_wk = get_pct("agy", "wk") or get_pct("agy", "week") or get_pct("agy", "gemini")
codex_wk = get_pct("codex", "gpt-6.2") or get_pct("codex", "wk") or get_pct("codex", "week")
fable_wk = get_pct("claude", "fable")

# Fallback parse from `usage --table --color never` if provider names differ
print(json.dumps({"gem_5h": gem_5h, "gem_wk": gem_wk, "codex_wk": codex_wk, "fable_wk": fable_wk}))
PY
)"

  USAGE_TABLE="$(usage --table --color never 2>/dev/null || true)"
  PARSED_PICK="$(python3 - "$USAGE_TABLE" "$SELECTION" <<'PY'
import json, re, sys

tbl = sys.argv[1]
sel = json.loads(sys.argv[2]) if sys.argv[2].startswith("{") else {}

def parse_table_pct(prefix: str) -> float | None:
    for line in tbl.splitlines():
        if line.strip().startswith(prefix):
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m:
                return float(m.group(1))
    return None

gem_5h = sel.get("gem_5h") if sel.get("gem_5h") is not None else parse_table_pct("Gemini 4 (5h)")
gem_wk = sel.get("gem_wk") if sel.get("gem_wk") is not None else parse_table_pct("Gemini 4 (Wk)")
codex_wk = sel.get("codex_wk") if sel.get("codex_wk") is not None else parse_table_pct("Codex GPT-6.2 (Wk)")
fable_wk = sel.get("fable_wk") if sel.get("fable_wk") is not None else parse_table_pct("Claude Fable 5.2 (Wk)")

gem_eff = min(gem_5h, gem_wk) if (gem_5h is not None and gem_wk is not None) else None

# Choose implementation agent: Gemini 4 Neon High or GPT-6.2-Astra xhigh (both >= 30% required)
# Alternate or pick higher effective quota so neither starves
candidates = []
if gem_eff is not None and gem_eff >= 30.0:
    candidates.append((gem_eff, "agy", "gemini-4-neon-high", "high", "gemini-4-neon-high"))
if codex_wk is not None and codex_wk >= 30.0:
    candidates.append((codex_wk, "codex", "gpt-6.2-astra", "xhigh", "gpt-6.2-astra/xhigh"))

candidates.sort(key=lambda x: x[0], reverse=True)
summary = f"Gemini4_5h={gem_5h}% Gemini4_Wk={gem_wk}% Codex_Wk={codex_wk}% Fable_Wk={fable_wk}%"
if not candidates:
    print(f"WAIT\tnone\tnone\tnone\tnone\t{summary}")
else:
    _, cli, model, effort, label = candidates[0]
    print(f"RUN\t{cli}\t{model}\t{effort}\t{label}\t{summary}")
PY
)"

  ACTION="$(printf '%s' "$PARSED_PICK" | cut -f1)"
  CLI_NAME="$(printf '%s' "$PARSED_PICK" | cut -f2)"
  MODEL_ID="$(printf '%s' "$PARSED_PICK" | cut -f3)"
  EFFORT_VAL="$(printf '%s' "$PARSED_PICK" | cut -f4)"
  AGENT_LABEL="$(printf '%s' "$PARSED_PICK" | cut -f5)"
  QUOTA_SUMMARY="$(printf '%s' "$PARSED_PICK" | cut -f6)"

  NOW_JST="$(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M')"
  if [[ "$ACTION" == "WAIT" ]]; then
    log "Both implementation quotas below 30% or unconfirmed ($QUOTA_SUMMARY). Sleeping 600s until quota resets..."
    sleep 600
    continue
  fi

  # Record sanitized allocation entry in canonical repo
  printf '| %s | %s R%d | trial-agent | %s | %s |\n' \
    "$NOW_JST" "$SITE" "$ROUND" "$AGENT_LABEL" "$QUOTA_SUMMARY" >> "$MAIN_REPO_DIR/records/allocation-log.md"

  PROMPT_TMP="$PRIVATE_DIR/round-prompt-${SITE}-r${ROUND}.md"
  sed "s/<site>/$SITE/g" "$LOOP_DIR/bench/round-prompt.md" > "$PROMPT_TMP"

  log "Starting round $ROUND on $SITE with $AGENT_LABEL ($QUOTA_SUMMARY)"
  set +e
  if [[ "$CLI_NAME" == "agy" ]]; then
    QFN_AGENT_MODEL="$AGENT_LABEL" "$AGENT_RUN_BIN" agy \
      -m "$MODEL_ID" \
      --access full \
      --no-update \
      -C "$LOOP_DIR" \
      --add-dir "$MAIN_REPO_DIR" \
      --add-dir "$PRIVATE_DIR" \
      --name "qfn-ar-${SITE}-r${ROUND}" \
      --prompt-file "$PROMPT_TMP"
    RC=$?
  else
    QFN_AGENT_MODEL="$AGENT_LABEL" "$AGENT_RUN_BIN" codex \
      -m "$MODEL_ID" \
      -e "$EFFORT_VAL" \
      --access full \
      --no-update \
      -C "$LOOP_DIR" \
      --add-dir "$MAIN_REPO_DIR" \
      --add-dir "$PRIVATE_DIR" \
      --name "qfn-ar-${SITE}-r${ROUND}" \
      --prompt-file "$PROMPT_TMP"
    RC=$?
  fi
  set -e

  log "Round $ROUND finished with exit code $RC"
  if [[ "$RC" == "3" ]]; then
    log "STOP: CLI reported unauthenticated / login required (exit 3)."
    exit 3
  fi

  ROUND=$(( ROUND + 1 ))
  sleep 10
done
