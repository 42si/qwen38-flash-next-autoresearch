# Quota & Agent Allocation Log (`records/allocation-log.md`)

Sanitized record of model roles and remaining quota percentages checked at the start of each round (`GPT-6.x-Sol` is never used for implementation, orchestration, or fallback; `Claude Fable 5.2` is never used for implementation).

| Checked (JST) | Round | Role | Model / Effort | Quota Snapshot |
|---|---|---|---|---|
| 2026-11-19 00:45 | setup / R1-init | implementation (repo, frozen `bench/`, loop launch) | Gemini 4 Carbon (High) | `Gemini4_5h=96% Gemini4_Wk=98% Codex_Wk=97% Fable_Wk=53% Cursor_Grok4.7=99% Colab_CU=2872.27` |
| 2026-11-19 00:45 | baseline (trial 0) | baseline measurement | GPT-6.2-Astra (xhigh) | `Codex_Wk=97%` |
| 2026-11-19 05:59 | spark R1 | trial-agent | gemini-4-carbon-high | Gemini4_5h=98.0% Gemini4_Wk=97.0% Codex_Wk=92.0% Fable_Wk=53.0% |
