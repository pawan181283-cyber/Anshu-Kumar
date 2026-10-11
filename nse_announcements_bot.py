"""
================================================================================
🧞‍♂️ Aladdin's Lamp — NSE Corporate Announcements Real-Time Intelligence & Telegram Alert Bot
(Production Ready: 24/7 Render Keep-Alive, Smart Filtering, Gemini AI Impact Analysis)
================================================================================

Polls NSE's own corporate-announcements feed directly (via the `nse` PyPI
package, which handles NSE's cookie/session requirements and rate limiting
for us), classifies each announcement's market impact using Gemini as a
Senior Equity Research Analyst persona, enriches with live LTP/% change via
NSE's own quote endpoint, and posts HTML-formatted Telegram alerts for
HIGH/MEDIUM impact announcements only.

DESIGN NOTES:

1. DATA SOURCE: uses the `nse` package (NSE.announcements() / NSE.quote())
   rather than hand-rolled requests + cookie scraping. NSE's site requires
   an initial cookie-establishing request before its JSON endpoints will
   respond, and blocks requests that skip this — a well-documented pain
   point. This package already solves it, with built-in rate limiting
   (3 req/s default) and retry-with-backoff on 429/502/503/504.

2. PRICE FIELDS (LTP / pChange) ARE NOT PART OF THE ANNOUNCEMENT ITSELF —
   NSE's announcements feed returns symbol/desc/an_dt/attchmntFile, not
   live price. This script fetches price separately via NSE.quote(symbol)
   for each announcement that survives dedup, decoupled from the Gemini
   classification call (which only needs the announcement text). The exact
   nested JSON path for LTP/pChange (priceInfo.lastPrice /
   priceInfo.pChange) is based on NSE's well-documented quote structure
   but has NOT been live-verified in this environment — see
   fetch_live_quote() below, which logs clearly and falls back to "N/A"
   rather than guessing if the structure doesn't match.

3. MODEL & DAILY BUDGET: uses the same resilient Gemini candidate list and
   daily call budget backstop established elsewhere in this project — see
   GEMINI_MODEL_CANDIDATES and GEMINI_DAILY_CALL_BUDGET.

4. TELEGRAM FORMAT: uses parse_mode="HTML" exactly as specified. Only &, <,
   > (and " inside the href attribute) need escaping — HTML mode doesn't
   have the single-vs-double-asterisk ambiguity that Telegram's "Markdown"
   mode does.

PRODUCTION / RENDER NOTES:

- The process binds 0.0.0.0:$PORT (default 10000) for Render's health check.
- A self-ping thread hits the health endpoint every 10 minutes to keep the
  free-tier instance awake.
- SIGTERM (Render's pre-shutdown signal) sets a shutdown event so the loop
  exits its current sleep promptly and closes the NSE session cleanly.
- The health endpoint returns 503 when no scan cycle has completed within
  2x the check interval, letting Render recycle a stuck instance.
- seen_announcements.json and gemini_usage.json live on the local
  ephemeral filesystem; they are recreated automatically after a restart.
- Announcements older than SEEN_PRUNE_AFTER_DAYS are pruned each cycle to
  bound memory and disk growth.

SETUP:
    pip install -r requirements.txt
    cp .env.example .env    # then fill in your real keys
    python nse_announcements_bot.py
================================================================================
"""

import os
import json
import time
import random
import signal
import hashlib
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone, timedelta

try:
    from dotenv import load_dotenv
except ImportError:
    os.system("pip install -q python-dotenv")
    from dotenv import load_dotenv

try:
    import requests
except ImportError:
    os.system("pip install -q requests")
    import requests

try:
    from nse import NSE
except ImportError:
    os.system("pip install -q nse")
    from nse import NSE

try:
    from google import genai
except ImportError:
    os.system("pip install -q google-genai")
    from google import genai

load_dotenv()


def log(msg: str) -> None:
    """Forced, immediately-flushed print so logs show up live wherever this runs."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================================
# 1. CONFIGURATION — loaded from .env via python-dotenv. No NSE API key is
# needed; the `nse` package authenticates via public session cookies only.
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RENDER_APP_URL = os.getenv("RENDER_APP_URL") or os.getenv("RENDER_EXTERNAL_URL", "")

PORT = int(os.getenv("PORT", "10000"))
SELF_PING_INTERVAL_SECONDS = 600

REQUEST_TIMEOUT = 20
# Not specified in the original request — 5 minutes is a reasonable default
# for an intraday announcement feed; tune via env var if you want faster
# or slower coverage.
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))

IST = timezone(timedelta(hours=5, minutes=30))

NSE_DATA_FOLDER = "nse_data"  # used by the `nse` package for its cookie file

GEMINI_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]
MODEL_NAME = None

GEMINI_MAX_RETRIES_PER_MODEL = 2
GEMINI_RETRY_BASE_DELAY = 2
GEMINI_RETRY_JITTER_SECONDS = 1.0
GEMINI_RESOLUTION_COOLDOWN_SECONDS = 300
_last_model_resolution_attempt = 0.0

MAX_EVALUATIONS_PER_CYCLE = int(os.getenv("MAX_EVALUATIONS_PER_CYCLE", "15"))
GEMINI_CALL_PACING_SECONDS = 2

GEMINI_DAILY_CALL_BUDGET = int(os.getenv("GEMINI_DAILY_CALL_BUDGET", "300"))
GEMINI_USAGE_PATH = "gemini_usage.json"

SEEN_ANNOUNCEMENTS_PATH = "seen_announcements.json"
SEEN_PRUNE_AFTER_DAYS = int(os.getenv("SEEN_PRUNE_AFTER_DAYS", "30"))

TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# ============================================================================
# 1b. GRACEFUL SHUTDOWN & LIVENESS STATE
# Render sends SIGTERM 30s before SIGKILL. We catch it, set an event, and let
# the main loop break out of its sleep so the NSE session closes cleanly.
# ============================================================================
_shutdown_event = threading.Event()
_last_successful_cycle_ts = 0.0


def _handle_shutdown_signal(signum, frame) -> None:
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)
    log(f"🛑 Received {sig_name} — initiating graceful shutdown...")
    _shutdown_event.set()


def _mark_cycle_healthy() -> None:
    global _last_successful_cycle_ts
    _last_successful_cycle_ts = time.time()


# ============================================================================
# 2. HEALTH-CHECK HTTP SERVER + SELF-PING (Render keep-alive)
# Returns 503 if no scan cycle finished within 2x the check interval (after
# an initial startup grace), so Render can recycle a wedged instance.
# ============================================================================
class _HealthCheckHandler(BaseHTTPRequestHandler):
    def _respond(self) -> None:
        now = time.time()
        startup_grace_seconds = 120
        max_staleness = max(CHECK_INTERVAL_SECONDS * 2, startup_grace_seconds)
        startup_time = getattr(self.server, "_bot_start_time", now)
        within_startup_grace = (now - startup_time) < startup_grace_seconds
        recently_completed = (
            _last_successful_cycle_ts > 0
            and (now - _last_successful_cycle_ts) < max_staleness
        )
        healthy = within_startup_grace or recently_completed

        if healthy:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"NSE Corporate Announcements Bot - running.")
        else:
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Retry-After", str(CHECK_INTERVAL_SECONDS))
            self.end_headers()
            self.wfile.write(b"NSE Corporate Announcements Bot - unhealthy (stale cycle).")

    def do_GET(self):
        self._respond()

    def do_HEAD(self):
        self._respond()

    def log_message(self, format, *args):
        pass


def start_health_check_server() -> None:
    def _serve():
        try:
            server = ThreadingHTTPServer(("0.0.0.0", PORT), _HealthCheckHandler)
            server._bot_start_time = time.time()
            log(f"✅ Health-check server listening on 0.0.0.0:{PORT}")
            server.serve_forever()
        except Exception as e:
            log(f"❌ Health-check server failed to start on port {PORT}: {e}")

    threading.Thread(target=_serve, daemon=True, name="health-check-server").start()


def start_self_ping_thread() -> None:
    if not RENDER_APP_URL:
        log("ℹ️ RENDER_APP_URL not set — self-ping disabled (expected outside Render).")
        return

    ping_url = RENDER_APP_URL.rstrip("/") + "/healthz"

    def _ping_loop():
        while not _shutdown_event.is_set():
            if _shutdown_event.wait(SELF_PING_INTERVAL_SECONDS):
                break
            try:
                r = requests.get(ping_url, timeout=10)
                log(f"🔁 Self-ping to {ping_url} -> HTTP {r.status_code}")
            except requests.exceptions.RequestException as e:
                log(f"⚠️ Self-ping failed (non-fatal): {e}")

    threading.Thread(target=_ping_loop, daemon=True, name="self-ping").start()
    log(f"✅ Self-ping thread started — pinging every {SELF_PING_INTERVAL_SECONDS}s.")


# ============================================================================
# 3. NSE CLIENT — instantiated once, kept alive for the whole process
# (NOT via a `with` block — that would close the session after one use,
# which is wrong for a long-running loop). Closed explicitly on shutdown.
# ============================================================================
nse_client = None


def init_nse_client() -> bool:
    global nse_client
    try:
        nse_client = NSE(download_folder=NSE_DATA_FOLDER)
        status = nse_client.status()
        log(f"✅ NSE client connected ({len(status)} market segment(s) reporting status).")
        return True
    except Exception as e:
        log(f"❌ Could not connect to NSE: {e}")
        return False


def shutdown_nse_client() -> None:
    global nse_client
    if nse_client is not None:
        try:
            nse_client.exit()
        except Exception:
            pass
        nse_client = None


def _ann_get(announcement, key: str, default=""):
    """Safe field accessor that works whether the `nse` package returns
    announcement items as dicts or as attribute-bearing objects."""
    if isinstance(announcement, dict):
        return announcement.get(key, default)
    return getattr(announcement, key, default)


# ============================================================================
# 4. STARTUP DIAGNOSTICS
# ============================================================================
def validate_config() -> bool:
    ok = True
    if not GEMINI_API_KEY:
        log("❌ GEMINI_API_KEY is missing. Set it in your .env file.")
        ok = False
    if not TELEGRAM_BOT_TOKEN or ":" not in TELEGRAM_BOT_TOKEN:
        log("❌ TELEGRAM_BOT_TOKEN is missing or looks invalid. Set it in your .env file.")
        ok = False
    if not TELEGRAM_CHAT_ID:
        log("❌ TELEGRAM_CHAT_ID is missing. Set it in your .env file.")
        ok = False
    return ok


def test_telegram_connection() -> bool:
    try:
        r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and r.json().get("ok"):
            log(f"✅ Telegram bot connected: @{r.json()['result'].get('username', 'unknown')}")
            return True
        log(f"❌ Telegram connection failed (HTTP {r.status_code}): {r.text[:300]}")
        return False
    except requests.exceptions.RequestException as e:
        log(f"❌ Telegram connection error: {e}")
        return False


def send_startup_notification() -> None:
    send_telegram_message(
        "🚀 <b>NSE Announcements Bot Connected!</b>\n"
        f"Classifying HIGH/MEDIUM impact announcements every {CHECK_INTERVAL_SECONDS // 60} min."
    )


# ============================================================================
# 5. GEMINI MODEL RESOLUTION (candidate fallback chain + daily budget)
# ============================================================================
def _today_utc_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_json_safe(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"⚠️ {path} corrupted or unreadable ({e}) — starting fresh.")
        return {}


def _save_json_atomic(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError as e:
        log(f"❌ Failed to save {path}: {e}")


def gemini_calls_used_today() -> int:
    return _load_json_safe(GEMINI_USAGE_PATH).get(_today_utc_str(), 0)


def record_gemini_call() -> None:
    usage = _load_json_safe(GEMINI_USAGE_PATH)
    today = _today_utc_str()
    usage[today] = usage.get(today, 0) + 1
    usage = {d: c for d, c in usage.items() if d == today}
    _save_json_atomic(GEMINI_USAGE_PATH, usage)


def gemini_budget_remaining() -> int:
    return max(0, GEMINI_DAILY_CALL_BUDGET - gemini_calls_used_today())


_genai_client = None


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def resolve_gemini_model(force: bool = False) -> bool:
    global MODEL_NAME, _last_model_resolution_attempt

    if MODEL_NAME is not None:
        return True

    now = time.time()
    if not force and (now - _last_model_resolution_attempt) < GEMINI_RESOLUTION_COOLDOWN_SECONDS:
        return False
    _last_model_resolution_attempt = now

    if gemini_budget_remaining() <= 0:
        log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) already used today. Skipping resolution.")
        return False

    try:
        client = get_genai_client()
    except Exception as e:
        log(f"❌ Could not create Gemini client: {e}")
        return False

    for candidate in GEMINI_MODEL_CANDIDATES:
        if gemini_budget_remaining() <= 0:
            break
        try:
            record_gemini_call()
            response = client.models.generate_content(model=candidate, contents="Reply with the single word: OK")
            text = (getattr(response, "text", "") or "").strip()
            MODEL_NAME = candidate
            log(f"✅ Gemini connected using model '{candidate}', test reply: '{text}'")
            return True
        except Exception as e:
            log(f"⚠️ Model '{candidate}' unavailable: {e}")
            continue

    log("❌ None of the candidate Gemini models responded right now.")
    log(f"   Will retry in ~{GEMINI_RESOLUTION_COOLDOWN_SECONDS}s. Check GEMINI_API_KEY and "
        f"https://ai.google.dev/gemini-api/docs/models for current model IDs.")
    return False


def _is_transient_gemini_error(e: Exception) -> bool:
    text = str(e).upper()
    return any(m in text for m in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "TIMEOUT", "DEADLINE"))


def _is_daily_quota_exhausted(e: Exception) -> bool:
    text = str(e).upper().replace("_", "")
    return "RESOURCEEXHAUSTED" in text and "PERDAY" in text


# ============================================================================
# 6. GEMINI IMPACT CLASSIFICATION — the persona + rules from the spec
# ============================================================================
GEMINI_PROMPT_TEMPLATE = """You are an expert Senior Indian Equity Research Analyst and Live News Automation Engine.

Analyze this raw NSE corporate announcement and classify its market impact.

HIGH IMPACT CRITERIA:
- Order Wins / Contract Awards (large deal sizes, government or global orders)
- Financial Results (QoQ / YoY Revenue & Profit growth spikes)
- Mergers, Acquisitions & Takeovers
- Capital Actions: Buybacks, Bonus Issues, Stock Splits, Special Dividends
- Negative Signals: Regulatory penalties, Fraud/Scam notices, Promoter defaults

MEDIUM IMPACT CRITERIA:
- Strategic Joint Ventures & Global Partnerships
- Plant Expansion, Capacity Addition, New Business Line Launch
- Major Credit Rating Upgrades or Downgrades
- Large Institutional Deals or Block Deals

LOW IMPACT — must be discarded:
- Loss of share certificates, routine Trading Window Closures, general
  Compliance filings, minor Director/Auditor changes, routine Postal
  Ballot notices, and any other routine/non-material disclosure.

If this announcement is LOW IMPACT, or doesn't clearly fit HIGH/MEDIUM, respond with EXACTLY the single word: DISCARD

If it IS HIGH or MEDIUM impact, respond in EXACTLY this structured format (plain text, no markdown, no HTML tags, one field per line, nothing else):

IMPACT: <HIGH or MEDIUM>
EVENT_TYPE: <short label, e.g. "Order Win / Contract Award", "Financial Results", "M&A", "Buyback", "Credit Rating Change">
SENTIMENT: <Bullish, Bearish, or Neutral>
SUMMARY: <exactly 1-2 crisp lines in professional Hinglish (Hindi+English mix) that a trader can read in 5 seconds. Always include the exact deal value or numeric impact if it is mentioned in the text below.>

---
SYMBOL: {symbol}
ANNOUNCEMENT SUBJECT/DESCRIPTION: {desc}
BROADCAST DATE & TIME: {an_dt}
---
"""


def parse_gemini_structured_response(text: str):
    """Returns a dict with impact/event_type/sentiment/summary, or None if
    the minimum usable fields (impact + summary) are missing — treated as
    malformed, not DISCARD, so it's retried rather than silently dropped."""
    fields = {}
    for line in text.strip().splitlines():
        line = line.strip()
        for key in ("IMPACT", "EVENT_TYPE", "SENTIMENT", "SUMMARY"):
            prefix = f"{key}:"
            if line.upper().startswith(prefix):
                fields[key.lower()] = line[len(prefix):].strip()
                break

    if not fields.get("impact") or not fields.get("summary"):
        return None
    if fields["impact"].strip().upper() not in ("HIGH", "MEDIUM"):
        return None  # defends against a LOW-impact item leaking into structured format

    fields.setdefault("event_type", "Corporate Announcement")
    fields.setdefault("sentiment", "Neutral")
    return fields


def analyze_announcement_with_gemini(announcement: dict):
    """Returns:
      - a parsed fields dict, for HIGH/MEDIUM impact announcements
      - the string "DISCARD" for LOW impact (safe to mark seen permanently)
      - None if every candidate model failed or the daily budget is
        exhausted (must NOT be marked seen — retry later)
    """
    global MODEL_NAME

    if not MODEL_NAME:
        return None
    if gemini_budget_remaining() <= 0:
        return None

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        symbol=_ann_get(announcement, "symbol", ""),
        desc=_ann_get(announcement, "desc", "") or "N/A",
        an_dt=_ann_get(announcement, "an_dt", ""),
    )
    client = get_genai_client()
    last_error = None

    models_to_try = [MODEL_NAME] + [m for m in GEMINI_MODEL_CANDIDATES if m != MODEL_NAME]

    for model_id in models_to_try:
        if gemini_budget_remaining() <= 0:
            break

        for attempt in range(1, GEMINI_MAX_RETRIES_PER_MODEL + 1):
            if gemini_budget_remaining() <= 0:
                return None

            try:
                record_gemini_call()
                response = client.models.generate_content(model=model_id, contents=prompt)
                text = (getattr(response, "text", "") or "").strip()

                if not text:
                    last_error = "empty response from model"
                    raise ValueError(last_error)

                if model_id != MODEL_NAME:
                    log(f"🔀 Falling back to model '{model_id}' (issues with '{MODEL_NAME}') — adopting as primary.")
                    MODEL_NAME = model_id

                if text.strip().upper() == "DISCARD":
                    return "DISCARD"

                parsed = parse_gemini_structured_response(text)
                if parsed is None:
                    last_error = f"unparseable structured response: {text[:200]!r}"
                    raise ValueError(last_error)

                return parsed

            except Exception as e:
                last_error = e

                if _is_daily_quota_exhausted(e):
                    log(f"⚠️ Daily quota exhausted for model '{model_id}': {e}")
                    if model_id == MODEL_NAME:
                        MODEL_NAME = None
                    break

                if _is_transient_gemini_error(e) and attempt < GEMINI_MAX_RETRIES_PER_MODEL:
                    delay = GEMINI_RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, GEMINI_RETRY_JITTER_SECONDS)
                    log(f"⚠️ Gemini transient error on attempt {attempt}/{GEMINI_MAX_RETRIES_PER_MODEL} "
                        f"for model '{model_id}': {e} — retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    log(f"⚠️ Model '{model_id}' failed ({e}) — trying next fallback candidate.")
                    break

    log(f"❌ All Gemini candidates failed for '{_ann_get(announcement, 'symbol', '')}': {last_error}")
    return None


# ============================================================================
# 7. LIVE QUOTE ENRICHMENT (LTP / % change via NSE's own quote endpoint)
# ============================================================================
def _as_dict(obj):
    """Normalize a value returned by the `nse` package (dict or attribute
    object) into a plain dict for safe .get() access."""
    if isinstance(obj, dict):
        return obj
    if obj is None:
        return {}
    try:
        return dict(obj)
    except (TypeError, ValueError):
        try:
            return {k: getattr(obj, k) for k in vars(obj)}
        except TypeError:
            return {}


def fetch_live_quote(symbol: str):
    """Returns (price, pChange) or (None, None) if unavailable. The nested
    path (priceInfo.lastPrice / priceInfo.pChange) matches NSE's
    well-documented quote structure but is NOT live-verified in this
    environment — if this starts returning (None, None) for every symbol,
    print one raw quote() response and check the actual key names here."""
    if nse_client is None:
        return None, None
    try:
        data = nse_client.quote(symbol)
        data = _as_dict(data)
        price_info = _as_dict(data.get("priceInfo", {}))
        price = price_info.get("lastPrice")
        pchange = price_info.get("pChange")
        if price is None or pchange is None:
            raise ValueError(f"priceInfo missing expected keys: {list(price_info.keys())}")
        return float(price), float(pchange)
    except Exception as e:
        log(f"⚠️ Live quote lookup failed for '{symbol}': {e}")
        return None, None


# ============================================================================
# 8. TIMESTAMP FORMATTING (IST) — NSE's an_dt is typically already IST-local
# ============================================================================
def format_ist_timestamp(an_dt: str) -> str:
    """NSE's an_dt field is already a local (IST) timestamp string in its
    own format, not a UNIX epoch — so this is a best-effort passthrough
    with light cleanup rather than a timezone conversion. Falls back to
    the raw value if nothing needs fixing."""
    if not an_dt:
        return datetime.now(IST).strftime("%d-%b-%Y %I:%M:%S %p IST")
    return str(an_dt).strip()


# ============================================================================
# 9. ANTI-DUPLICATE MEMORY (seen_announcements.json)
# ============================================================================
def load_seen() -> dict:
    return _load_json_safe(SEEN_ANNOUNCEMENTS_PATH)


def save_seen(seen: dict) -> None:
    _save_json_atomic(SEEN_ANNOUNCEMENTS_PATH, seen)


def prune_seen(seen: dict) -> int:
    """Drop entries older than SEEN_PRUNE_AFTER_DAYS so the seen-store
    doesn't grow without bound over months of 24/7 operation."""
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=SEEN_PRUNE_AFTER_DAYS)).isoformat()
    before = len(seen)
    kept = {}
    for ann_id, meta in seen.items():
        if not isinstance(meta, dict):
            kept[ann_id] = meta
            continue
        seen_at = meta.get("seen_at")
        if not seen_at or seen_at >= cutoff_iso:
            kept[ann_id] = meta
    seen.clear()
    seen.update(kept)
    removed = before - len(kept)
    if removed:
        log(f"🧹 Pruned {removed} stale seen announcement(s) older than {SEEN_PRUNE_AFTER_DAYS} days.")
    return removed


def make_announcement_id(announcement: dict) -> str:
    """NSE's announcement items don't reliably expose one confirmed unique
    ID field across all index types, so this hashes the combination of
    symbol + broadcast time + description + attachment link — together
    these are effectively unique per real announcement."""
    raw = "|".join(
        [
            str(_ann_get(announcement, "symbol", "")),
            str(_ann_get(announcement, "an_dt", "")),
            str(_ann_get(announcement, "desc", "")),
            str(_ann_get(announcement, "attchmntFile", "")),
        ]
    ).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def mark_seen(seen: dict, ann_id: str, symbol: str, status: str) -> None:
    seen[ann_id] = {
        "symbol": symbol,
        "status": status,
        "seen_at": datetime.now(timezone.utc).isoformat(),
    }
    save_seen(seen)


# ============================================================================
# 10. NSE ANNOUNCEMENTS INGESTION
# ============================================================================
def fetch_nse_announcements(max_retries: int = 2) -> list:
    """The `nse` package already retries on 429/502/503/504 internally
    (see RetryConfig in its docs), so this wraps only for errors outside
    that — e.g. a connection drop during our own process's lifetime.
    If all retries fail, the NSE session cookie may have expired, so we
    attempt one client reinitialization before giving up for this cycle."""
    global nse_client
    if nse_client is None:
        return []

    for attempt in range(1, max_retries + 1):
        try:
            announcements = nse_client.announcements(index="equities")
            if not isinstance(announcements, list):
                log(f"⚠️ Unexpected NSE announcements response shape ({type(announcements).__name__}).")
                return []
            log(f"Fetched {len(announcements)} announcement(s) from NSE.")
            return announcements
        except Exception as e:
            delay = (2 ** attempt) + random.uniform(0, 1)
            log(f"⚠️ NSE announcements fetch error (attempt {attempt}/{max_retries}): {e} — retrying in {delay:.1f}s")
            time.sleep(delay)

    log("🔄 NSE fetch failed after retries — reinitializing client (session may have expired)...")
    shutdown_nse_client()
    if not init_nse_client():
        log("❌ NSE client reinitialization failed; skipping fetch for this cycle.")
        return []

    try:
        announcements = nse_client.announcements(index="equities")
        if isinstance(announcements, list):
            log(f"Fetched {len(announcements)} announcement(s) from NSE after reinit.")
            return announcements
        log(f"⚠️ Unexpected NSE response shape after reinit ({type(announcements).__name__}).")
    except Exception as e:
        log(f"❌ NSE fetch failed even after client reinit: {e}")

    return []


# ============================================================================
# 11. TELEGRAM FORMATTING & DISPATCH — HTML mode, exactly as specified
# ============================================================================
IMPACT_DISPLAY = {
    "HIGH": "🔥 HIGH IMPACT",
    "MEDIUM": "⚡ MEDIUM IMPACT",
}

SENTIMENT_DISPLAY = {
    "BULLISH": "🟢 Bullish",
    "BEARISH": "🔴 Bearish",
    "NEUTRAL": "⚪ Neutral",
}


def escape_html(text) -> str:
    """Only &, <, > need escaping in Telegram's HTML mode — far simpler
    and more predictable than Markdown's escaping rules."""
    if text is None:
        return ""
    text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_html_attr(text) -> str:
    """Same as escape_html, plus quote-escaping for safe use inside an
    href="..." attribute."""
    return escape_html(text).replace('"', "&quot;")


def format_price_display(price, pchange) -> str:
    if price is None or pchange is None:
        return "N/A"
    sign = "+" if pchange >= 0 else ""
    return f"{price:,.2f} ({sign}{pchange:.2f}%)"


def format_telegram_message(announcement: dict, fields: dict, price, pchange, published_str: str) -> str:
    symbol = escape_html(_ann_get(announcement, "symbol", ""))
    impact = fields["impact"].strip().upper()
    impact_display = IMPACT_DISPLAY.get(impact, f"⚡ {impact} IMPACT")
    sentiment_display = SENTIMENT_DISPLAY.get(fields.get("sentiment", "Neutral").strip().upper(), "⚪ Neutral")
    event_type = escape_html(fields.get("event_type", "Corporate Announcement"))
    summary = escape_html(fields["summary"])
    price_display = format_price_display(price, pchange)
    attachment_url = escape_html_attr(_ann_get(announcement, "attchmntFile", ""))

    doc_line = ""
    if _ann_get(announcement, "attchmntFile", ""):
        doc_line = f'\n\n📄 <b>Official Document:</b> <a href="{attachment_url}">View Official NSE PDF</a>'

    return (
        f"🚨 <b>NSE LIVE ALERT | {symbol}</b> 🚨\n\n"
        f"📌 <b>Stock Symbol:</b> <code>{symbol}</code>\n"
        f"💰 <b>Current Price (LTP):</b> ₹{price_display}\n"
        f"⏱️ <b>Broadcast Time:</b> {escape_html(published_str)}\n\n"
        f"📢 <b>Event Type:</b> {event_type}\n"
        f"💥 <b>Impact Level:</b> {impact_display}\n"
        f"💡 <b>Market Sentiment:</b> {sentiment_display}\n\n"
        f"📝 <b>Summary:</b>\n{summary}"
        f"{doc_line}"
    )


def send_telegram_message(text: str, max_retries: int = 3) -> bool:
    # Telegram's hard limit is 4096 UTF-16 code units; truncate defensively
    # so an unusually long summary never causes the whole send to fail.
    if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 6] + "\n…"
        log(f"⚠️ Telegram message truncated to {TELEGRAM_MAX_MESSAGE_LENGTH} chars.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                delay = (2 ** attempt) + random.uniform(0, 1)
                log(f"⚠️ Telegram rate-limited (429). Retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            log(f"❌ Telegram HTTP {r.status_code}: {r.text[:300]}")
            return False
        except requests.exceptions.RequestException as e:
            delay = (2 ** attempt) + random.uniform(0, 1)
            log(f"⚠️ Telegram request error (attempt {attempt}/{max_retries}): {e} — retrying in {delay:.1f}s")
            time.sleep(delay)

    return False


# ============================================================================
# MAIN CYCLE + CONTINUOUS LOOP
# ============================================================================
def run_cycle() -> None:
    log("--- Starting scan cycle ---")

    if not MODEL_NAME:
        if not resolve_gemini_model():
            log("⏳ Gemini still unavailable this cycle (cooldown, outage, or budget exhausted). Skipping scan.")
            return

    announcements = fetch_nse_announcements()
    log(f"Total announcements fetched: {len(announcements)}")

    seen = load_seen()
    prune_seen(seen)
    remaining_budget = gemini_budget_remaining()
    log(f"Gemini budget: {remaining_budget}/{GEMINI_DAILY_CALL_BUDGET} left today, "
        f"MAX_EVALUATIONS_PER_CYCLE={MAX_EVALUATIONS_PER_CYCLE}.")

    items_evaluated = 0
    posted_count = 0

    for announcement in announcements:
        if _shutdown_event.is_set():
            log("🛑 Shutdown requested mid-cycle — stopping.")
            break

        try:
            ann_id = make_announcement_id(announcement)
        except Exception as e:
            log(f"⚠️ Could not compute announcement id, skipping: {e}")
            continue

        if ann_id in seen:
            continue

        symbol = _ann_get(announcement, "symbol", "")

        if items_evaluated >= MAX_EVALUATIONS_PER_CYCLE:
            log(f"⏭️ MAX_EVALUATIONS_PER_CYCLE ({MAX_EVALUATIONS_PER_CYCLE}) reached — "
                f"remaining announcements picked up next cycle.")
            break

        if gemini_budget_remaining() <= 0:
            log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) exhausted mid-cycle. Stopping.")
            break

        result = analyze_announcement_with_gemini(announcement)
        time.sleep(GEMINI_CALL_PACING_SECONDS)

        if result is None:
            log(f"⏭️ Leaving unprocessed for retry next cycle: {symbol}")
            continue

        items_evaluated += 1

        if result == "DISCARD":
            mark_seen(seen, ann_id, symbol, "discarded_low_impact")
            continue

        fields = result
        price, pchange = fetch_live_quote(symbol)
        published_str = format_ist_timestamp(_ann_get(announcement, "an_dt", ""))
        message = format_telegram_message(announcement, fields, price, pchange, published_str)

        if send_telegram_message(message):
            mark_seen(seen, ann_id, symbol, "posted")
            posted_count += 1
            log(f"✅ Alert sent [{fields['impact'].upper()}]: {symbol}")
        else:
            log(f"⚠️ Delivery failed, will retry next cycle: {symbol}")

        time.sleep(1.5)

    log(f"--- Cycle complete: {items_evaluated} evaluated by Gemini, {posted_count} alert(s) sent, "
        f"{gemini_calls_used_today()}/{GEMINI_DAILY_CALL_BUDGET} Gemini calls used today ---")


def main() -> None:
    log("=" * 60)
    log(" 🧞‍♂️ Aladdin's Lamp — NSE Announcements Bot — starting up ")
    log("=" * 60)

    # Register graceful-shutdown handlers for Render (SIGTERM) and Ctrl-C (SIGINT).
    try:
        signal.signal(signal.SIGTERM, _handle_shutdown_signal)
        signal.signal(signal.SIGINT, _handle_shutdown_signal)
    except (ValueError, OSError) as e:
        # signal() can only be called from the main thread; harmless if it fails.
        log(f"⚠️ Could not install signal handlers: {e}")

    start_health_check_server()
    start_self_ping_thread()

    if not validate_config():
        log("🛑 Fix the .env values above and restart. Stopping now.")
        return

    if not test_telegram_connection():
        log("🛑 Telegram connection failed — nothing works without this. Stopping.")
        return

    if not init_nse_client():
        log("🛑 Could not establish an NSE session — nothing works without this. Stopping.")
        return

    if not resolve_gemini_model(force=True):
        log("⚠️ Gemini is not reachable right now (see errors above). Starting the loop anyway — "
            "it will keep retrying in the background.")

    try:
        send_startup_notification()
    except Exception:
        log("⚠️ Startup notification failed (non-fatal), continuing.")

    log("✅ Startup checks complete. Entering continuous monitoring loop.")

    try:
        while not _shutdown_event.is_set():
            try:
                run_cycle()
            except Exception:
                log("❌ Unhandled error in monitoring cycle:")
                traceback.print_exc()

            _mark_cycle_healthy()

            if _shutdown_event.is_set():
                break

            log(f"Sleeping {CHECK_INTERVAL_SECONDS}s until next scan...")
            _shutdown_event.wait(CHECK_INTERVAL_SECONDS)
    finally:
        log("🧹 Shutting down NSE client and exiting.")
        shutdown_nse_client()


if __name__ == "__main__":
    main()
