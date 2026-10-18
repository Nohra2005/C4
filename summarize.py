"""Rebuild a clean results table from logs/experiments.jsonl.

For each setup it keeps the LATEST complete run (all 10 test cases) and ignores API errors
(quota / server timeouts are not the agent's fault). Writes logs/results_final.csv.
Usage: python summarize.py
"""
import csv
import json

from experiments import CASES, SETUPS

N = len(CASES)
rows = [json.loads(l) for l in open("logs/experiments.jsonl", encoding="utf-8")]

# Split the log into runs: consecutive rows of the same setup, starting at the first test case
runs, current = [], []
for r in rows:
    if r["case"] == CASES[0]["id"] and current:
        runs.append(current)
        current = []
    current.append(r)
if current:
    runs.append(current)

latest = {}
for run in runs:
    if len(run) == N and len({r["setup"] for r in run}) == 1:
        latest[run[0]["setup"]] = run  # later runs overwrite earlier ones

table = []
for name in SETUPS:
    if name not in latest:
        continue
    run = latest[name]
    errors = [r for r in run if r["reply"].startswith("ERROR:")]
    ok = [r for r in run if not r["reply"].startswith("ERROR:")]
    n = max(len(ok), 1)
    table.append({
        "setup": name,
        "api_errors": len(errors),
        "pass_rate": round(sum(r["pass"] for r in ok) / n, 2),
        "tools_ok": round(sum(r["tools_ok"] for r in ok) / n, 2),
        "facts_ok": round(sum(r["facts_ok"] for r in ok) / n, 2),
        "no_hallucination": round(sum(r["no_hallucination"] for r in ok) / n, 2),
        "format_errors": sum(r["format_errors"] for r in ok),
        "avg_steps": round(sum(r["steps"] for r in ok) / n, 1),
        "avg_words": round(sum(r["words"] for r in ok) / n),
        "avg_latency_s": round(sum(r["latency_s"] for r in ok) / n, 1),
        "failed_cases": ", ".join(r["case"] for r in ok if not r["pass"]),
    })

with open("logs/results_final.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

for t in table:
    print(f"{t['setup']:<16} pass={t['pass_rate']:<4} fmt_err={t['format_errors']} "
          f"latency={t['avg_latency_s']}s  failed: {t['failed_cases'] or '-'}")
print("Saved logs/results_final.csv")
