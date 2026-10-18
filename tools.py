"""OpenEar tools for the ReAct agent.

All facts come from the OpenEar business files (me/about_business.pdf), turned into Python data so the
agent has to *look things up* through tools instead of reading everything from its prompt.
Every tool returns a dict; the ReAct loop turns it into an Observation.
"""
import json
import os
import re
from datetime import datetime

# ---------- Business data (from me/about_business.pdf) ----------
CATALOG = {
    "stride": {
        "name": "OpenEar Stride", "price": 79, "conduction": "air", "battery_h": 8, "ip": "IP55 (sweat and rain)",
        "weight_g": 28, "colors": ["black", "grey"],
        "best_for": ["casual running", "everyday running", "gym"],
        "notes": "Entry model. Air conduction gives fuller bass than bone conduction. Not made for swimming.",
    },
    "pace pro": {
        "name": "OpenEar Pace Pro", "price": 129, "conduction": "bone", "battery_h": 10, "ip": "IP67",
        "weight_g": 26, "colors": ["black", "red", "blue"],
        "best_for": ["serious running", "trail running", "cycling", "marathon"],
        "notes": "Most popular model. Secure wraparound fit, dual microphones for calls.",
    },
    "swim": {
        "name": "OpenEar Swim", "price": 159, "conduction": "bone", "battery_h": 8, "ip": "IP68 up to 2 m",
        "weight_g": None, "colors": None,
        "best_for": ["swimming", "triathlon", "open water"],
        "notes": "32 GB built-in MP3 storage, because Bluetooth does not work underwater. "
                 "Best with the earplugs from the Sport Fit Kit.",
    },
    "ultra": {
        "name": "OpenEar Ultra", "price": 189, "conduction": "bone", "battery_h": 16, "ip": "IP67",
        "weight_g": None, "colors": None,
        "best_for": ["ultra running", "long-distance", "hiking", "technical terrain"],
        "notes": "Quick charge: 10 minutes gives 2 hours. Ear-hook lock for rocky terrain.",
    },
    "sport fit kit": {
        "name": "Sport Fit Kit", "price": 15, "conduction": None, "battery_h": None, "ip": None,
        "weight_g": None, "colors": None, "best_for": ["accessory, fits every model"],
        "notes": "Spare silicone hooks in two sizes, swimming earplugs, zip carry pouch.",
    },
}

# Sport keyword -> recommended model (FAQ: "Which model should I choose?")
SPORT_MAP = {
    "stride": ["casual", "jog", "everyday", "beginner", "gym", "walk"],
    "pace pro": ["serious", "trail", "cycl", "bike", "marathon", "half", "road", "10k", "race", "running", "run"],
    "swim": ["swim", "pool", "triathlon", "open water"],
    "ultra": ["ultra", "long-distance", "long distance", "100k", "50k", "hik", "technical"],
}

POLICIES = {
    "delivery": "Beirut and Mount Lebanon: within 24 hours. Rest of Lebanon: 2 to 3 days. "
                "Fee $4, free on orders over $100. Only delivers inside Lebanon. "
                "If a customer asks for delivery outside Lebanon, log it with record_feedback.",
    "payment": "Cash on delivery or card. Never take card details in the chat.",
    "warranty": "1-year warranty against manufacturing defects on every headphone.",
    "returns": "Unused items in original packaging can be returned within 14 days of delivery for a full refund.",
    "contact": "Email support@openear.store or DM @openear.lb on Instagram. The team replies within one working day.",
    "charging": "Every model charges with the included magnetic USB cable.",
    "helmet": "All models sit in front of the ear and work with cycling helmets and sport sunglasses.",
    "calls": "Only the Pace Pro has dual microphones for calls.",
    "testing": "Every model is tested by the team on real runs, rides and swims before it is sold.",
    "team": "Founded by Rami Haddad (Head of Product, marathon runner) and Lea Khoury "
            "(Head of Operations, triathlete).",
}

FAST_ZONE = ["beirut", "achrafieh", "hamra", "verdun", "mar mikhael", "gemmayze", "badaro", "ras beirut",
             "jounieh", "baabda", "aley", "broummana", "beit mery", "antelias", "dbayeh", "jal el dib", "zalka",
             "hazmieh", "metn", "chouf", "kesrouan", "keserwan", "jbeil", "byblos", "mount lebanon", "dekwaneh",
             "sin el fil", "bikfaya", "kaslik", "zouk", "naccache", "mansourieh", "choueifat", "damour"]
REST_OF_LEBANON = ["tripoli", "saida", "sidon", "tyre", "sour", "zahle", "baalbek", "nabatieh", "batroun",
                   "chekka", "akkar", "halba", "jezzine", "marjayoun", "bint jbeil", "hermel", "koura",
                   "zgharta", "ehden", "bekaa", "chtaura", "anjar", "rachaya", "north lebanon", "south lebanon"]


def _find_product(name):
    key = (name or "").lower().replace("openear", "").strip()
    for k in CATALOG:
        if k == key or k.replace(" ", "") == key.replace(" ", "") or k in key:
            return k
    return None


# ---------- Lookup tools ----------
def search_catalog(sport: str = "", budget: float = None):
    """Recommend models for a sport, optionally under a budget."""
    sport_l = (sport or "").lower()
    recommended = [k for k, words in SPORT_MAP.items() if any(w in sport_l for w in words)]
    # longest match wins (e.g. "ultra running" -> ultra, not pace pro)
    if "ultra" in recommended or "swim" in recommended:
        recommended = [k for k in recommended if k in ("ultra", "swim")]
    elif "stride" in recommended:
        recommended = ["stride"]
    candidates = recommended or [k for k in CATALOG if k != "sport fit kit"]
    results = []
    for k in candidates:
        p = CATALOG[k]
        if budget is not None and p["price"] > float(budget):
            continue
        results.append({"name": p["name"], "price": p["price"], "best_for": p["best_for"], "colors": p["colors"]})
    out = {"sport": sport, "budget": budget, "matches": results}
    if not results:
        cheapest = min((CATALOG[k] for k in candidates), key=lambda p: p["price"])
        out["note"] = f"Nothing fits that budget. The recommended model for this sport is {cheapest['name']} " \
                      f"at ${cheapest['price']}."
    if not recommended:
        out["note"] = out.get("note", "") + " Sport not recognised: showing all models; ask the customer which sport."
    return out


def get_product(name: str):
    """Full specs, price and colour options for one product."""
    k = _find_product(name)
    if not k:
        return {"error": f"Unknown product '{name}'. Products: Stride, Pace Pro, Swim, Ultra, Sport Fit Kit."}
    p = dict(CATALOG[k])
    if p["colors"] is None and k != "sport fit kit":
        p["colors"] = "not listed in the business files - do not invent; the team confirms colour by email"
    return p


def compare_products(a: str, b: str):
    """Side-by-side comparison of two products."""
    ka, kb = _find_product(a), _find_product(b)
    if not ka or not kb:
        return {"error": "Unknown product. Products: Stride, Pace Pro, Swim, Ultra, Sport Fit Kit."}
    fields = ["price", "conduction", "battery_h", "ip", "weight_g", "colors", "best_for", "notes"]
    return {f: {CATALOG[ka]["name"]: CATALOG[ka][f], CATALOG[kb]["name"]: CATALOG[kb][f]} for f in fields}


def delivery_quote(city: str, cart_total: float):
    """Delivery fee, total and delivery time for a city in Lebanon."""
    c = (city or "").lower().strip()
    total = float(cart_total)
    fee = 0 if total > 100 else 4
    if any(z in c for z in FAST_ZONE):
        eta = "within 24 hours"
    elif any(z in c for z in REST_OF_LEBANON):
        eta = "2 to 3 days"
    else:
        return {"status": "unknown_city",
                "note": f"'{city}' is not a known city in Lebanon. OpenEar only delivers inside Lebanon. "
                        "If it is outside Lebanon, call record_feedback; if unsure, ask the customer."}
    return {"city": city, "cart_total": total, "delivery_fee": fee, "grand_total": total + fee, "eta": eta}


def get_policy(topic: str):
    """Store policy for a topic: delivery, payment, warranty, returns, contact, charging, helmet, calls, testing, team."""
    t = (topic or "").lower()
    for k, v in POLICIES.items():
        if k in t or t in k:
            return {"topic": k, "policy": v}
    return {"error": f"No policy on '{topic}'. Known topics: {', '.join(POLICIES)}. "
                     "If the customer needs this, call record_feedback."}


# ---------- Action tools (from C3, colour added) ----------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DRY_RUN = False  # experiments set this to True so test leads don't pollute the real log


def _append_log(filename, record):
    if DRY_RUN:
        return
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(LOG_DIR, filename), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_customer_interest(name: str, email: str, product: str, color: str = "", message: str = ""):
    """Save a lead. Validates the email and, for models with colour options, the colour."""
    if not EMAIL_RE.match((email or "").strip()):
        return {"status": "error", "reason": "Invalid email address. Ask the customer to re-enter it."}
    k = _find_product(product)
    if not k:
        return {"status": "error", "reason": f"Unknown product '{product}'."}
    colors = CATALOG[k]["colors"]
    if colors:
        if not color or color.lower().strip() not in colors:
            return {"status": "error",
                    "reason": f"Colour missing or not available. Ask the customer to choose: {', '.join(colors)}."}
    _append_log("leads.jsonl", {"name": name, "email": email, "product": CATALOG[k]["name"],
                                "color": color or "to confirm", "message": message})
    return {"status": "recorded", "product": CATALOG[k]["name"], "color": color or "to be confirmed by the team"}


def record_feedback(question: str):
    """Log a question the files cannot answer, a complaint or a suggestion."""
    _append_log("feedback.jsonl", {"question": question})
    return {"status": "recorded", "type": "feedback",
            "tell_customer": "The team will follow up. Contact: support@openear.store or @openear.lb on Instagram."}


TOOLS = {
    "search_catalog": search_catalog,
    "get_product": get_product,
    "compare_products": compare_products,
    "delivery_quote": delivery_quote,
    "get_policy": get_policy,
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
}

# Embedded in the system prompt (the model never sees Python signatures, only this text)
TOOL_DESCRIPTIONS = """
- search_catalog: recommend models for a sport. Input: {"sport": str, "budget": number or null}
- get_product: specs, price and colour options of one product. Input: {"name": str}
- compare_products: compare two products side by side. Input: {"a": str, "b": str}
- delivery_quote: delivery fee, grand total and delivery time. Input: {"city": str, "cart_total": number}
- get_policy: store policy. Input: {"topic": one of delivery, payment, warranty, returns, contact, charging,
  helmet, calls, testing, team}
- record_customer_interest: save an order request. Input: {"name": str, "email": str, "product": str,
  "color": str, "message": str}. Only call it once you have name, email, product AND colour
  (colour is required for Stride and Pace Pro).
- record_feedback: log a question the tools cannot answer, a complaint or a suggestion. Input: {"question": str}
""".strip()
