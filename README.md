# OpenEar ReAct Agent with Configurable Personas (C4)

**EECE 503P/798S Agentic Systems** · Tatiana Nohra

This project builds on my C3 business agent ([openear](https://github.com/Nohra2005/openear)): Maguy, the sales
assistant of **OpenEar**, a fictional Lebanese store for open-ear sports headphones.

In C3, Gemini decided on tool calls through its built-in function calling. In C4 I wrote the **ReAct loop
myself in LangGraph**. The model writes `Thought / Action / Action Input` as plain text, my code parses it,
runs the tool and feeds back an `Observation`, until the model writes a `Final Answer`.

## How it works

```
route → reason → parse ──action──→ act ──→ reason      (loop, max 6 tool calls)
                       ──final───→ finish
                       ──invalid─→ fix ──→ reason      (format error sent back to the model)
                       ──limit───→ fallback
```

- **route:** the C3 router, now choosing a persona for each message
- **reason:** one Gemini call; stops at `Observation:` so the model can't invent tool results
- **parse:** detects the action (regex + JSON) or the final answer; guards reject an answer that claims
  "recorded" without calling a record tool, or talks about delivery without checking a delivery tool
- **act:** runs the Python tool and returns the observation
- No prebuilt agent (`create_react_agent`, `ToolNode`, `AgentExecutor`) is used.

## Personas

| Persona | Task | Style | Config |
|---|---|---|---|
| Coach Maguy | pick a model for the customer's sport | warm, sporty, ends with a question | temp 0.8, top-p 0.95, top-k 40 |
| Ziad, Tech Specialist | specs and comparisons | precise, bullet points, exact numbers | temp 0.1, top-p 0.5, top-k 5 |
| Nour, Order Desk | collect product, **colour**, name, email, city; record the lead | careful, one question at a time | temp 0.2, top-p 0.8, top-k 20 |

Presence / frequency penalties are part of the persona design, but every available Gemini model rejects them,
so `llm.py` drops them and logs it.

## Tools

`search_catalog`, `get_product`, `compare_products`, `delivery_quote`, `get_policy`,
`record_customer_interest` (validates email **and colour**), `record_feedback`.

## Results (12 setups × 10 test questions)

| Setup | Pass rate | Avg latency |
|---|---|---|
| **baseline** (router + CoT + zero-shot + system prompt + gemini-3.1-flash-lite) | **1.0** | 11 s |
| no CoT | 0.75 | 16 s |
| few-shot | 1.0 | 26 s |
| instructions in the user message | 1.0 | 31 s |
| no instructions block | 0.8 | 35 s |
| temperature 0 / 1.3 | 0.9 / 1.0 | 17 s / 12 s |
| top-p 0.3, top-k 3 | 0.8 | 20 s |
| gemini-3.5-flash | 1.0 | 40 s |
| Coach / Tech / Order Desk only | 1.0 / 1.0 / 0.9 | 5 s / 9 s / 10 s |

Full table: `logs/results_final.csv` · every run: `logs/experiments.jsonl` · screenshots: `screenshots/` ·
analysis and reflection: `c4_react_agent.ipynb`.

## Files

| File | Purpose |
|---|---|
| `c4_react_agent.ipynb` | Use case, personas, instructions, ReAct graph, live demo, results and reflection |
| `react_agent.py` | The LangGraph ReAct loop (router, parser, guards, tool execution); run it to chat in the terminal |
| `tools.py` | Product catalog, policies and the 7 tools |
| `personas.py` | The 3 personas: prompt traits, LLM config and why each setting was chosen |
| `prompts.py` | Output format, instructions block (with worked example), few-shot examples, prompt builder |
| `llm.py` | Gemini wrapper: model fallback, retries, 30 s timeout, drops unsupported parameters |
| `experiments.py` | Runs the 10 test questions under the 12 setups and logs the results |
| `summarize.py` | Builds the clean results table from the experiment log |
| `me/` | OpenEar business files (from C3) |

## Run locally

```bash
pip install -r requirements.txt
# create a .env file containing: GEMINI_API_KEY=your_key
python react_agent.py          # chat with the agent in the terminal
python experiments.py          # run all experiments (about 400 API calls)
python summarize.py            # rebuild logs/results_final.csv
```
