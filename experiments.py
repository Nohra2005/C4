"""Experiment harness: run the same test questions under different setups and log the results.

Usage:
    python experiments.py                  # all setups
    python experiments.py baseline no_cot  # only some setups

Results: logs/experiments.jsonl (one line per run) and logs/experiments_summary.csv (one line per setup).
"""
import csv
import json
import os
import sys

import tools
from react_agent import run_agent

tools.DRY_RUN = True  # test leads / feedback are not written to the real logs

# ---------- Test cases ----------
# expect_tools: must be called | forbid_tools: must NOT be called
# must_include: words the reply must contain | must_not_include: words that would mean a hallucination
CASES = [
    {"id": "swim", "message": "I swim and do triathlons, which headphones should I get?",
     "expect_tools": ["search_catalog"], "must_include": ["swim"]},
    {"id": "mtb_budget", "message": "I do mountain biking, my budget is $140. What do you recommend?",
     "expect_tools": ["search_catalog"], "must_include": ["pace pro", "129"]},
    {"id": "compare", "message": "Which lasts longer on battery, the Ultra or the Pace Pro?",
     "expect_tools": ["compare_products"], "must_include": ["16", "10"]},
    {"id": "stride_swim", "message": "Can I swim with the Stride?",
     "expect_tools": ["get_product"], "must_include": ["stride"]},
    {"id": "delivery", "message": "How much is delivery to Tripoli if I buy a Stride, and how long does it take?",
     "expect_tools": ["delivery_quote"], "must_include": ["4", "2"]},
    {"id": "dubai", "message": "Do you ship to Dubai?",
     "expect_tools": ["record_feedback"], "must_include": ["support@openear.store"]},
    {"id": "ask_colour", "message": "I want the Pace Pro. I'm Rima, rima@gmail.com",
     "forbid_tools": ["record_customer_interest"], "must_include": ["black", "red", "blue"]},
    {"id": "order_done", "message": "Blue please. I live in Beirut.",
     "history": [{"role": "user", "content": "I want the Pace Pro. I'm Karim, karim@gmail.com"},
                 {"role": "assistant", "content": "Great choice Karim! Which colour: black, red or blue?"}],
     "expect_tools": ["record_customer_interest"], "must_include": ["blue"]},
    {"id": "swim_colours", "message": "What colours does the Swim come in?",
     "expect_tools": ["get_product"],
     "must_not_include": ["black", "grey", "gray", "white", "red", "blue", "yellow", "pink", "green"]},
    {"id": "off_topic", "message": "Write me a poem about Beirut.",
     "forbid_tools": ["record_customer_interest"], "must_not_include": ["poem:"]},
]

# ---------- Setups: change ONE thing at a time vs the baseline ----------
BASE = {"persona": "auto", "settings": {"cot": True, "few_shot": False, "placement": "system",
                                        "use_instructions": True}}


def variant(**changes):
    s = {**BASE["settings"], **{k: v for k, v in changes.items() if k != "persona"}}
    return {"persona": changes.get("persona", "auto"), "settings": s}


SETUPS = {
    "baseline": variant(),
    "no_cot": variant(cot=False),
    "few_shot": variant(few_shot=True),
    "in_message": variant(placement="message"),
    "no_instructions": variant(use_instructions=False),
    "temp_0": variant(config={"temperature": 0.0}),
    "temp_1.3": variant(config={"temperature": 1.3}),
    "top_p_0.3": variant(config={"top_p": 0.3, "top_k": 3}),
    "model_3.5_flash": variant(model="gemini-3.5-flash"),
    "only_coach": variant(persona="coach"),
    "only_tech": variant(persona="tech"),
    "only_order": variant(persona="order"),
}


def score(case, out):
    reply, used = out["final"].lower(), set(out["tools_used"])
    tools_ok = set(case.get("expect_tools", [])) <= used and not (set(case.get("forbid_tools", [])) & used)
    facts_ok = all(w in reply for w in case.get("must_include", []))
    no_halluc = not any(w in reply for w in case.get("must_not_include", []))
    return {"tools_ok": tools_ok, "facts_ok": facts_ok, "no_hallucination": no_halluc,
            "pass": tools_ok and facts_ok and no_halluc}


def run(setup_names):
    os.makedirs("logs", exist_ok=True)
    summary = []
    for name in setup_names:
        setup = SETUPS[name]
        rows = []
        for case in CASES:
            try:
                out = run_agent(case["message"], case.get("history", []), persona=setup["persona"],
                                settings=setup["settings"])
            except Exception as e:  # one failed API call shouldn't kill the whole experiment
                out = {"final": f"ERROR: {e}", "persona": setup["persona"], "tools_used": [], "steps": 0,
                       "format_errors": 0, "latency_s": 0, "llm_calls": 0, "trace": []}
            m = score(case, out)
            m["api_error"] = out["final"].startswith("ERROR:")
            row = {"setup": name, "case": case["id"], "persona": out["persona"], "tools_used": out["tools_used"],
                   "steps": out["steps"], "format_errors": out["format_errors"], "latency_s": out["latency_s"],
                   "llm_calls": out["llm_calls"], "words": len(out["final"].split()), **m, "reply": out["final"]}
            rows.append(row)
            with open("logs/experiments.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = "API ERROR" if m["api_error"] else ("PASS" if m["pass"] else "FAIL")
            print(f"[{name}] {case['id']:<13} {status}  tools={out['tools_used']}")
        errors = sum(r["api_error"] for r in rows)
        rows = [r for r in rows if not r["api_error"]]  # API errors are not the agent's fault: leave them out
        n = max(len(rows), 1)
        summary.append({
            "setup": name, "api_errors": errors,
            "pass_rate": round(sum(r["pass"] for r in rows) / n, 2),
            "tools_ok": round(sum(r["tools_ok"] for r in rows) / n, 2),
            "facts_ok": round(sum(r["facts_ok"] for r in rows) / n, 2),
            "no_hallucination": round(sum(r["no_hallucination"] for r in rows) / n, 2),
            "format_errors": sum(r["format_errors"] for r in rows),
            "avg_steps": round(sum(r["steps"] for r in rows) / n, 1),
            "avg_words": round(sum(r["words"] for r in rows) / n),
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 1),
        })
        print(summary[-1], "\n")
    path = "logs/experiments_summary.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        if new:
            w.writeheader()
        w.writerows(summary)
    print(f"Saved {path}")


if __name__ == "__main__":
    run(sys.argv[1:] or list(SETUPS))
