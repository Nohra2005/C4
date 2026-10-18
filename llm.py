"""Gemini wrapper: model fallback on overload / rate limit (from C3), and dropping parameters a model rejects."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv(override=True)
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Put it in .env (or as a Space secret on Hugging Face).")
client = genai.Client(api_key=_api_key, http_options=types.HttpOptions(timeout=30000))  # 30 s max per call

DEFAULT_MODEL = "gemini-3.1-flash-lite"
# 3.5-flash-lite often returns an empty MALFORMED_RESPONSE on ReAct-style prompts, so it is not the default
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
PENALTIES = ("presence_penalty", "frequency_penalty")
_penalty_unsupported = set()  # models that rejected penalties once; we stop sending them


def call_llm(contents, system=None, config=None, model=None, stop=None, retries=3):
    """One Gemini call. Returns (text, info) where info logs model, dropped params, latency and tokens."""
    cfg = dict(config or {})
    models = [model] if model else []
    models += [m for m in FALLBACK_MODELS if m not in models]
    last_error, dropped, malformed = None, [], []
    for m in models:
        params = dict(cfg)
        if m in _penalty_unsupported:
            dropped = [p for p in PENALTIES if params.pop(p, None) is not None]
        for attempt in range(retries):
            gen_cfg = types.GenerateContentConfig(
                system_instruction=system, stop_sequences=stop,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),  # visible ReAct thoughts only
                **params)
            t0 = time.time()
            try:
                r = client.models.generate_content(model=m, contents=contents, config=gen_cfg)
                reason = str(r.candidates[0].finish_reason) if r.candidates else ""
                if not r.text and "MALFORMED" in reason:
                    malformed.append(m)
                    break  # this model can't produce the format: try the next model
                usage = r.usage_metadata
                return (r.text or ""), {
                    "model": m, "dropped_params": dropped, "malformed_models": malformed, "latency_s": round(time.time() - t0, 2),
                    "in_tokens": getattr(usage, "prompt_token_count", None),
                    "out_tokens": getattr(usage, "candidates_token_count", None)}
            except (errors.ClientError, errors.ServerError) as e:
                code, msg = getattr(e, "code", None), str(e)
                if code == 400 and "Penalty is not enabled" in msg:
                    _penalty_unsupported.add(m)
                    dropped = [p for p in PENALTIES if params.pop(p, None) is not None]
                    continue
                if code not in (429, 500, 503, 504):
                    raise
                last_error = e
                if code == 429 and attempt == retries - 1:
                    break  # quota for this model is used up: try the next one
                time.sleep(5 * (attempt + 1) if code == 429 else 2 * (attempt + 1))
            except Exception as e:  # network timeout (our 30 s limit): retry, then try the next model
                if "timed out" not in str(e).lower() and "timeout" not in type(e).__name__.lower():
                    raise
                last_error = e
                time.sleep(2)
    if last_error:
        raise last_error
    return "", {"model": None, "dropped_params": dropped, "malformed_models": malformed, "latency_s": 0,
                "in_tokens": None, "out_tokens": None}
