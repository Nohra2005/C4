"""Manual ReAct loop for OpenEar, built as a LangGraph state machine (no prebuilt agent / ToolNode).

Graph:
  route -> reason -> parse --action--> act --> reason        (loop)
                          --final---> finish
                          --invalid-> fix ---> reason        (format error fed back as an Observation)
                          --limit---> fallback
"""
import json
import re
import time
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, StateGraph

from llm import DEFAULT_MODEL, call_llm
from personas import PERSONAS, ROUTE_TO_PERSONA
from prompts import build_prompt
from tools import TOOLS

MAX_STEPS = 6          # max tool calls per customer message
MAX_FORMAT_ERRORS = 2  # unparseable outputs tolerated before giving up
FALLBACK_REPLY = "Sorry, I couldn't sort that out. Please contact support@openear.store or @openear.lb on Instagram."


class AgentState(TypedDict, total=False):
    message: str          # latest customer message
    history: list         # [{"role": "user"|"assistant", "content": str}]
    persona: str          # "auto", "coach", "tech" or "order"
    settings: dict        # cot, few_shot, placement, use_instructions, model, config
    scratchpad: str       # Thought / Action / Observation text for this message
    last_output: str
    parsed: dict
    steps: int
    format_errors: int
    tools_used: list
    trace: list
    final: str


# ---------- Router (C3's router, now choosing a persona) ----------
ROUTER_PROMPT = """Classify the customer's LATEST message for OpenEar, a store for open-ear sports headphones.
- recommend: choosing a model for their sport, or general questions
- specs: technical specs, battery, waterproofing, comparing models
- order: wants to buy or order, gives name / email / colour / city for an order, asks delivery cost of an order
- other: anything else
Recent conversation:
{recent}
Latest message: {message}
Reply with one word: recommend, specs, order or other."""


def route_node(state):
    if state["persona"] != "auto":
        return {}
    recent = "".join(f"{t['role']}: {t['content']}\n" for t in state["history"][-2:]) or "(none)"
    text, info = call_llm(ROUTER_PROMPT.format(recent=recent, message=state["message"]),
                          config={"temperature": 0}, model=state["settings"].get("model"))
    label = next((k for k in ROUTE_TO_PERSONA if k in text.lower()), "other")
    persona = ROUTE_TO_PERSONA[label]
    return {"persona": persona, "trace": state["trace"] + [{"node": "route", "label": label, "persona": persona}]}


# ---------- Reason: one LLM step ----------
def _contents(state, prefix):
    contents = [types.Content(role="model" if t["role"] == "assistant" else "user",
                              parts=[types.Part.from_text(text=t["content"])]) for t in state["history"]]
    turn = f"{prefix}Customer: {state['message']}\n\n{state['scratchpad']}"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn)]))
    return contents


def reason_node(state):
    s = state["settings"]
    persona = PERSONAS[state["persona"]]
    system, prefix = build_prompt(persona, cot=s.get("cot", True), few_shot=s.get("few_shot", False),
                                  placement=s.get("placement", "system"),
                                  use_instructions=s.get("use_instructions", True))
    config = {**persona["config"], **s.get("config", {})}
    # Stop before the model invents its own Observation: the tool result must come from our code
    text, info = call_llm(_contents(state, prefix), system=system, config=config,
                          model=s.get("model", DEFAULT_MODEL), stop=["Observation:"])
    text = text.strip()
    return {"last_output": text, "scratchpad": state["scratchpad"] + text + "\n",
            "trace": state["trace"] + [{"node": "reason", "output": text, **info}]}


# ---------- Parse: detect an action or a final answer ----------
ACTION_RE = re.compile(r"^\s*Action\s*:\s*([A-Za-z_]+)", re.I | re.M)
INPUT_RE = re.compile(r"Action\s*Input\s*:\s*", re.I)
FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*", re.I)
INLINE_RE = re.compile(r"^\s*Action\s*:\s*([A-Za-z_]+)\s*\((.*)\)\s*$", re.I | re.M)


def _json_after(text, pos):
    """Decode the first JSON object that starts at or after pos."""
    start = text.find("{", pos)
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_react(text):
    """Return {"kind": "action", "tool", "args"} | {"kind": "final", "answer"} | {"kind": "invalid", "error"}."""
    action, final = ACTION_RE.search(text), FINAL_RE.search(text)
    thought = re.search(r"Thought\s*:\s*(.*?)(?=\n\s*(Action|Final\s*Answer)\s*:|\Z)", text, re.S | re.I)
    thought = thought.group(1).strip() if thought else ""
    # Whichever comes first wins (a model may write an Action and then hallucinate a Final Answer)
    if action and (not final or action.start() < final.start()):
        tool = action.group(1)
        if tool not in TOOLS:
            return {"kind": "invalid", "error": f"Unknown tool '{tool}'. Available tools: {', '.join(TOOLS)}."}
        inp = INPUT_RE.search(text, action.end())
        args = _json_after(text, inp.end()) if inp else None
        if args is None:  # also accept inline style: Action: get_product({"name": "Swim"}) or get_product("Swim")
            inline = INLINE_RE.search(text)
            if inline:
                raw = inline.group(2).strip()
                args = _json_after(raw, 0) if raw.startswith("{") else {"__positional__": raw.strip("'\" ")}
        if args is None:
            return {"kind": "invalid", "error": "Action Input must be a JSON object, e.g. {\"name\": \"Swim\"}."}
        return {"kind": "action", "tool": tool, "args": args, "thought": thought}
    if final:
        answer = text[final.end():].strip()
        if answer:
            return {"kind": "final", "answer": answer, "thought": thought}
    if text.strip() and "Thought:" not in text:   # plain reply with no ReAct format = final answer
        return {"kind": "final", "answer": text.strip(), "thought": ""}
    return {"kind": "invalid", "error": "No Action or Final Answer found. Use the required output format."}


def parse_node(state):
    parsed = parse_react(state["last_output"])
    # Guard: the agent may not claim it recorded something without actually calling the tool
    if parsed["kind"] == "final" and re.search(r"\b(recorded|logged)\b", parsed["answer"], re.I) \
            and not {"record_customer_interest", "record_feedback"} & set(state["tools_used"]):
        parsed = {"kind": "invalid", "error": "You told the customer you recorded something, but you did not "
                  "call record_customer_interest or record_feedback. Call the tool first, or don't claim it."}
    # Guard: no delivery / shipping answer without checking a delivery tool first
    elif parsed["kind"] == "final" and re.search(r"\b(ship|shipping|deliver|delivery)\b", parsed["answer"], re.I) \
            and not {"get_policy", "delivery_quote"} & set(state["tools_used"]):
        parsed = {"kind": "invalid", "error": "You answered about delivery without checking. Call get_policy "
                  "(topic: delivery) or delivery_quote first."}
    return {"parsed": parsed, "trace": state["trace"] + [{"node": "parse", **parsed}]}

# ---------- Act: run the tool, return the observation ----------
def act_node(state):
    p = state["parsed"]
    func, args = TOOLS[p["tool"]], dict(p["args"])
    try:
        if "__positional__" in args:
            result = func(args.pop("__positional__"))
        else:
            result = func(**args)
    except TypeError as e:  # wrong argument names
        result = {"error": f"Bad input for {p['tool']}: {e}"}
    obs = json.dumps(result, ensure_ascii=False)
    return {"scratchpad": state["scratchpad"] + f"Observation: {obs}\n",
            "steps": state["steps"] + 1, "tools_used": state["tools_used"] + [p["tool"]],
            "trace": state["trace"] + [{"node": "act", "tool": p["tool"], "args": p["args"], "observation": result}]}


def fix_node(state):
    err = state["parsed"]["error"]
    return {"scratchpad": state["scratchpad"] + f"Observation: Format error: {err}\n",
            "format_errors": state["format_errors"] + 1,
            "trace": state["trace"] + [{"node": "fix", "error": err}]}


def finish_node(state):
    return {"final": state["parsed"]["answer"]}


def fallback_node(state):
    # If the model just replied in plain text (no Action), use that text rather than a canned apology
    last = state.get("last_output", "")
    reply = last if last and not ACTION_RE.search(last) and "Thought:" not in last else FALLBACK_REPLY
    return {"final": reply, "trace": state["trace"] + [{"node": "fallback", "used_plain_text": reply != FALLBACK_REPLY}]}


def after_parse(state):
    kind = state["parsed"]["kind"]
    if kind == "final":
        return "finish"
    if kind == "action":
        return "act" if state["steps"] < MAX_STEPS else "fallback"
    return "fix" if state["format_errors"] < MAX_FORMAT_ERRORS else "fallback"


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("route", route_node), ("reason", reason_node), ("parse", parse_node), ("act", act_node),
                     ("fix", fix_node), ("finish", finish_node), ("fallback", fallback_node)]:
        g.add_node(name, fn)
    g.set_entry_point("route")
    g.add_edge("route", "reason")
    g.add_edge("reason", "parse")
    g.add_conditional_edges("parse", after_parse,
                            {"act": "act", "finish": "finish", "fix": "fix", "fallback": "fallback"})
    g.add_edge("act", "reason")
    g.add_edge("fix", "reason")
    g.add_edge("finish", END)
    g.add_edge("fallback", END)
    return g.compile()


GRAPH = build_graph()


def run_agent(message, history=None, persona="auto", settings=None, verbose=False):
    """Run one customer turn. Returns final reply, persona used, tools used and the full trace."""
    t0 = time.time()
    state = GRAPH.invoke({"message": message, "history": history or [], "persona": persona,
                          "settings": settings or {}, "scratchpad": "", "steps": 0, "format_errors": 0,
                          "tools_used": [], "trace": []}, {"recursion_limit": 50})
    if verbose:
        print_trace(state["trace"])
    return {"final": state["final"], "persona": state["persona"], "tools_used": state["tools_used"],
            "steps": state["steps"], "format_errors": state["format_errors"], "trace": state["trace"],
            "latency_s": round(time.time() - t0, 2),
            "llm_calls": sum(1 for t in state["trace"] if t["node"] in ("reason", "route"))}


def print_trace(trace):
    for t in trace:
        if t["node"] == "route":
            print(f"[ROUTE] {t['label']} -> {PERSONAS[t['persona']]['name']}")
        elif t["node"] == "reason":
            print(f"[LLM {t['model']}]\n{t['output']}")
        elif t["node"] == "act":
            print(f"Observation: {json.dumps(t['observation'], ensure_ascii=False)}")
        elif t["node"] == "fix":
            print(f"[FORMAT ERROR] {t['error']}")
        elif t["node"] == "fallback":
            print("[FALLBACK]")


if __name__ == "__main__":
    history = []
    while True:
        msg = input("You: ")
        if msg.lower() in ("quit", "exit"):
            break
        out = run_agent(msg, history, verbose=True)
        print(f"\n{PERSONAS[out['persona']]['name']}: {out['final']}\n")
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": out["final"]}]
