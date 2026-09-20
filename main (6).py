"""
================================================================================
AlphaRadar AI - Indian Stock Market Real-Time News Filter & Telegram Alert Bot
(Fixed build: fail-fast startup diagnostics, forced print() logging, real loop)
================================================================================
Run this in a single Google Colab cell. To stop it, interrupt the cell
(Runtime > Interrupt execution) — it's designed to run forever otherwise.

News source: scraped from Zerodha Pulse (pulse.zerodha.com), an aggregator
of major Indian financial news publishers — replaces the earlier
direct-RSS approach (Economic Times / Moneycontrol / Livemint).

COLAB SETUP (run once, in the cell above this one):
    !pip install -q beautifulsoup4 requests google-genai
"""

import os
import re
import sys
import time
import random
import sqlite3
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone

try:
    from bs4 import BeautifulSoup
except ImportError:
    os.system("pip install -q beautifulsoup4")
    from bs4 import BeautifulSoup

try:
    import requests
except ImportError:
    os.system("pip install -q requests")
    import requests

try:
    from google import genai
except ImportError:
    os.system("pip install -q google-genai")
    from google import genai


def log(msg: str) -> None:
    """Forced, immediately-flushed print so Colab shows live output even
    inside tight loops or right before a crash."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================================
# CONFIGURATION — pulled from environment variables (Render dashboard),
# never hardcoded. Each accepts either naming convention so it matches
# whatever you've actually set in Render's Environment tab:
#   TELEGRAM_BOT_TOKEN / BOT_TOKEN
#   TELEGRAM_CHAT_ID   / CHAT_ID
#   GEMINI_API_KEY
# ============================================================================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Render's Web Service tier requires binding to a port within its scan
# timeout, or the deploy is killed even though the bot itself works fine.
# This doesn't serve real traffic — it exists purely to satisfy that check.
PORT = int(os.getenv("PORT", "10000"))

# Render's free tier spins down a Web Service after 15 minutes with no
# INBOUND HTTP traffic — and wipes its (ephemeral) local filesystem when it
# does, which means news_history.db and the daily Gemini call counter get
# silently reset too. Self-pinging our own public URL is a best-effort
# backup, NOT a guaranteed fix (Render's own docs call it a workaround) —
# the reliable fix is an external pinger (cron-job.org, UptimeRobot, etc.)
# hitting this URL every <15 minutes. Set BOTH up for redundancy.
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # auto-set by Render for Web Services
SELF_PING_INTERVAL_SECONDS = 600  # 10 min — safely under the 15-min spin-down window

DB_PATH = "news_history.db"

# Google retires/blocks specific Gemini model IDs with little notice
# (e.g. gemini-2.5-flash was blocked for new API keys ahead of its official
# retirement). Rather than hardcoding one version, we try a short candidate
# list at startup and lock onto whichever one actually works.
# "gemini-flash-latest" is Google's alias that auto-points at the current
# Flash model, so this list should keep working across future model bumps.
GEMINI_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]
MODEL_NAME = None  # resolved at startup by resolve_gemini_model()

# How often the bot scans RSS feeds. This alone does NOT cap Gemini usage —
# a single busy cycle can still contain many fresh headlines. It's paired
# with GEMINI_DAILY_CALL_BUDGET below, which is the actual quota guardrail.
CHECK_INTERVAL_SECONDS = 900  # 15 minutes — matches Zerodha Pulse's near-real-time updates.
REQUEST_TIMEOUT = 20

# Retry settings for transient Gemini errors (503 UNAVAILABLE, 429 rate
# limit, timeouts). These are temporary server-side conditions, not real
# failures, so we retry the SAME model a couple of times with exponential
# backoff + jitter before falling back to the NEXT candidate model
# (see GEMINI_MODEL_CANDIDATES) entirely — a 503 from high demand on one
# model doesn't mean another model is also overloaded.
GEMINI_MAX_RETRIES_PER_MODEL = 2  # attempts on one model before falling back
GEMINI_RETRY_BASE_DELAY = 2  # seconds — doubles each attempt: 2s, 4s...
GEMINI_RETRY_JITTER_SECONDS = 1.0  # up to +1s random jitter, avoids thundering-herd retries

# ----------------------------------------------------------------------
# QUOTA GUARDRAIL — the actual fix for hitting daily rate limits.
# Every real Gemini call (analysis attempts AND model-probe calls) counts
# against this budget, tracked per calendar day (UTC) in SQLite so it
# survives restarts. Set this comfortably below your real daily quota —
# free tier is commonly ~20/day per model; with up to 4 candidate models
# in the list above, 40 is a conservative shared budget. Tune to match
# whatever ai.google.dev/gemini-api/docs/rate-limits shows for your key.
# ----------------------------------------------------------------------
GEMINI_DAILY_CALL_BUDGET = 40

# Minimum number of items *evaluated* (decisively — success or genuine
# IGNORE, not failures) per cycle, even when 40/24-style even-spread math
# would round down to 1 and leave a whole hour vulnerable to a single
# transient 503 eating the entire cycle. The daily budget check remains
# the hard safety ceiling regardless of this value — raising it only
# changes how many items get a fair shot per cycle, not total spend.
GEMINI_MIN_ITEMS_PER_CYCLE = 5  # try 10 if you want more coverage per hour

# How often we're willing to re-probe candidate models when Gemini is down.
# Prevents burning quota by retrying on every restart or every scan cycle —
# each probe call costs against the budget just like a real analysis call.
GEMINI_RESOLUTION_COOLDOWN_SECONDS = 300
_last_model_resolution_attempt = 0.0

# Zerodha Pulse aggregates major Indian financial publishers (Economic
# Times, NDTV Profit, The Hindu Business, Business Standard, etc.) into one
# near-real-time feed — replaces hitting each publisher's RSS directly,
# which reduces both latency and duplicate coverage of the same story.
ZERODHA_PULSE_URL = "https://pulse.zerodha.com/"
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================================
# HEALTH-CHECK HTTP SERVER (for Render Web Service port binding)
# Serves a trivial 200 OK on 0.0.0.0:PORT so Render's deploy port-scan
# passes. Runs in a background daemon thread — never blocks or interferes
# with the bot's own polling loop, which keeps running on the main thread.
# ============================================================================
class _HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"AlphaRadar AI - Bot is running.")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress http.server's default per-request logging — Render's
        # own port-scan and any uptime monitor would otherwise spam the logs.
        pass


def start_health_check_server() -> None:
    """Starts the health-check server on a daemon thread. Called first,
    before config validation or any network calls, so the port opens as
    fast as possible and isn't delayed by (or dependent on) Telegram/Gemini
    connectivity checks."""
    def _serve():
        try:
            server = HTTPServer(("0.0.0.0", PORT), _HealthCheckHandler)
            log(f"✅ Health-check server listening on 0.0.0.0:{PORT}")
            server.serve_forever()
        except Exception as e:
            log(f"❌ Health-check server failed to start on port {PORT}: {e}")

    thread = threading.Thread(target=_serve, daemon=True, name="health-check-server")
    thread.start()


def start_self_ping_thread() -> None:
    """Best-effort backup to keep the free-tier service from spinning down:
    pings our own public URL every SELF_PING_INTERVAL_SECONDS. This is NOT
    a substitute for an external pinger (cron-job.org, UptimeRobot) — set
    one of those up too. RENDER_EXTERNAL_URL is populated automatically by
    Render for Web Services; if it's empty (e.g. running locally, or on a
    Background Worker where this doesn't apply), self-ping is skipped."""
    if not RENDER_EXTERNAL_URL:
        log("ℹ️ RENDER_EXTERNAL_URL not set — self-ping disabled (expected when running locally). "
            "On Render, set up an external ping to this service's URL every <15 min "
            "(cron-job.org or UptimeRobot, both free) so it never spins down.")
        return

    ping_url = RENDER_EXTERNAL_URL.rstrip("/") + "/"

    def _ping_loop():
        while True:
            time.sleep(SELF_PING_INTERVAL_SECONDS)
            try:
                r = requests.get(ping_url, timeout=10)
                log(f"🔁 Self-ping to {ping_url} -> HTTP {r.status_code}")
            except requests.exceptions.RequestException as e:
                log(f"⚠️ Self-ping failed (non-fatal, will retry next interval): {e}")

    thread = threading.Thread(target=_ping_loop, daemon=True, name="self-ping")
    thread.start()
    log(f"✅ Self-ping thread started — pinging {ping_url} every {SELF_PING_INTERVAL_SECONDS}s. "
        f"Still set up an external pinger too — this alone isn't guaranteed reliable.")


# ============================================================================
# STARTUP DIAGNOSTICS — catches placeholder tokens & bad keys BEFORE the loop
# so you get a clear message instead of a buried 404
# ============================================================================
def validate_config() -> bool:
    ok = True

    if not BOT_TOKEN or BOT_TOKEN.startswith("YOUR_") or ":" not in BOT_TOKEN:
        log("❌ BOT_TOKEN is empty or invalid. Check that TELEGRAM_BOT_TOKEN (or BOT_TOKEN) is set "
            "in Render's Environment tab and looks like '123456789:AAE...'.")
        ok = False

    if not CHAT_ID or CHAT_ID.startswith("YOUR_"):
        log("❌ CHAT_ID is empty or invalid. Check that TELEGRAM_CHAT_ID (or CHAT_ID) is set "
            "in Render's Environment tab. Use @userinfobot or getUpdates to find your chat id.")
        ok = False

    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("YOUR_"):
        log("❌ GEMINI_API_KEY is empty or invalid. Check that it's set in Render's Environment tab.")
        ok = False

    return ok


def test_telegram_connection() -> bool:
    """Calls getMe — the cheapest possible Telegram call — so a bad token
    surfaces immediately as a readable message instead of a 404 later."""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and r.json().get("ok"):
            bot_name = r.json()["result"].get("username", "unknown")
            log(f"✅ Telegram bot connected: @{bot_name}")
            return True
        log(f"❌ Telegram connection failed (HTTP {r.status_code}): {r.text[:300]}")
        log("   This is the classic 404 cause: BOT_TOKEN is wrong or was never set.")
        return False
    except requests.exceptions.RequestException as e:
        log(f"❌ Telegram connection error: {e}")
        return False


def send_startup_notification() -> bool:
    """Fires a plain confirmation message to Telegram once, at startup,
    so you know end-to-end delivery works before any real alerts depend on it."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🚀 AlphaRadar AI Connected Successfully! Monitoring Started...",
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            log("✅ Startup notification sent to Telegram.")
            return True
        log(f"⚠️ Startup notification failed (HTTP {r.status_code}): {r.text[:300]}")
        return False
    except requests.exceptions.RequestException as e:
        log(f"⚠️ Startup notification error: {e}")
        return False


def resolve_gemini_model(force: bool = False) -> bool:
    """Tries each candidate model in order and locks onto the first one
    that actually responds. Resilient to two separate failure modes:
      - a model being deprecated/retired (404) -> just skip to the next one
      - a model's quota being exhausted (RESOURCE_EXHAUSTED) -> also skip to
        the next one, since free-tier quotas are typically tracked per
        (project, model), so a different candidate may still have budget.

    Cooldown-limited: repeated calls within GEMINI_RESOLUTION_COOLDOWN_SECONDS
    are skipped (no API calls made, returns whatever the last known state
    was) so a Gemini outage doesn't turn into a quota-burning probe loop —
    whether from Render restarts or from every single poll cycle retrying.
    Pass force=True to bypass the cooldown (used once at startup).
    """
    global MODEL_NAME, _last_model_resolution_attempt

    if MODEL_NAME is not None:
        return True

    now = time.time()
    if not force and (now - _last_model_resolution_attempt) < GEMINI_RESOLUTION_COOLDOWN_SECONDS:
        return False
    _last_model_resolution_attempt = now

    remaining = gemini_budget_remaining()
    if remaining <= 0:
        log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) already used today. "
            f"Skipping model resolution — resumes automatically after midnight UTC.")
        return False

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        log(f"❌ Could not create Gemini client: {e}")
        return False

    for candidate in GEMINI_MODEL_CANDIDATES:
        if gemini_budget_remaining() <= 0:
            log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) exhausted mid-probe. "
                f"Stopping resolution attempts for today.")
            break
        try:
            record_gemini_call()
            response = client.models.generate_content(
                model=candidate,
                contents="Reply with the single word: OK",
            )
            text = (getattr(response, "text", "") or "").strip()
            MODEL_NAME = candidate
            log(f"✅ Gemini connected using model '{candidate}', test reply: '{text}' "
                f"(budget used today: {gemini_calls_used_today()}/{GEMINI_DAILY_CALL_BUDGET})")
            return True
        except Exception as e:
            log(f"⚠️ Model '{candidate}' unavailable: {e}")
            continue

    log("❌ None of the candidate Gemini models responded right now.")
    log("   Common causes: invalid GEMINI_API_KEY, outdated google-genai package, all candidates "
        "deprecated, or free-tier daily quota exhausted across every candidate.")
    log(f"   Will automatically retry in ~{GEMINI_RESOLUTION_COOLDOWN_SECONDS}s. "
        f"Check https://ai.google.dev/gemini-api/docs/models for current model IDs and "
        f"https://ai.google.dev/gemini-api/docs/rate-limits for your quota.")
    return False


# ============================================================================
# DATABASE LAYER (SQLite deduplication)
# ============================================================================
def init_db() -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_news (
                news_id      TEXT PRIMARY KEY,
                title        TEXT,
                source       TEXT,
                status       TEXT,
                processed_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gemini_usage (
                usage_date  TEXT PRIMARY KEY,
                calls_used  INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()
        log("✅ Database ready.")
    except sqlite3.DatabaseError as e:
        log(f"⚠️ Database file corrupted ({e}). Recreating it.")
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def gemini_calls_used_today() -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT calls_used FROM gemini_usage WHERE usage_date = ?", (_today_utc_str(),))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 0
    except sqlite3.Error as e:
        log(f"DB read error (gemini_usage): {e}")
        return 0


def record_gemini_call() -> None:
    """Increments today's Gemini call counter. Call this once for every
    real API request made — probe calls during model resolution included,
    since those cost against the quota exactly like an analysis call does."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        today = _today_utc_str()
        cur.execute(
            """
            INSERT INTO gemini_usage (usage_date, calls_used) VALUES (?, 1)
            ON CONFLICT(usage_date) DO UPDATE SET calls_used = calls_used + 1
            """,
            (today,),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log(f"DB write error (gemini_usage): {e}")


def gemini_budget_remaining() -> int:
    return max(0, GEMINI_DAILY_CALL_BUDGET - gemini_calls_used_today())


def is_news_processed(news_id: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sent_news WHERE news_id = ?", (news_id,))
        res = cur.fetchone()
        conn.close()
        return res is not None
    except sqlite3.Error as e:
        log(f"DB read error: {e}")
        return False


def mark_news_processed(news_id: str, title: str, source: str, status: str) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO sent_news (news_id, title, source, status, processed_at) VALUES (?, ?, ?, ?, ?)",
            (news_id, title, source, status, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log(f"DB write error: {e}")


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    return re.sub(r"\s+", " ", raw).strip()


# ============================================================================
# 1. NEWS INGESTION — scraped from Zerodha Pulse (pulse.zerodha.com)
# ============================================================================
def fetch_all_feeds() -> list:
    """Scrapes the 'ul#news li' items off Zerodha Pulse. Returns the same
    shape the rest of the pipeline already expects: a list of
    {id, title, summary, link, source} dicts — so Gemini analysis, dedup,
    and Telegram formatting all work completely unchanged. 'source' is the
    original publisher (e.g. 'NDTV Business', 'The Hindu Business') when
    Pulse's markup includes it, since Pulse itself is an aggregator, not
    the original author of any given story.
    """
    all_entries = []

    try:
        resp = requests.get(ZERODHA_PULSE_URL, headers=SCRAPE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log(f"❌ Failed to fetch Zerodha Pulse: {e}")
        return all_entries

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        news_list = soup.select_one("ul#news")
        if news_list is None:
            log("⚠️ Could not find 'ul#news' on Zerodha Pulse — the page structure may have "
                "changed. Fetched 0 entries this cycle.")
            return all_entries

        items = news_list.find_all("li", recursive=False)
        for li in items:
            try:
                link_tag = li.find("a", href=True)
                if not link_tag:
                    continue

                title = clean_text(link_tag.get_text())
                link = link_tag["href"].strip()
                if not title or not link:
                    continue

                # Best-effort extraction of publication source: Pulse
                # typically renders "X minutes/hours ago — Publisher Name"
                # as trailing text in each <li>. Falls back gracefully if
                # the markup doesn't match exactly.
                li_text = clean_text(li.get_text(" "))
                source_match = re.search(r"—\s*([A-Za-z0-9&.,' ]+?)\s*$", li_text)
                source = source_match.group(1).strip() if source_match else "Zerodha Pulse"

                desc_tag = li.find(["p", "div"], class_=re.compile("desc|summary", re.I))
                summary = clean_text(desc_tag.get_text()) if desc_tag else ""

                all_entries.append(
                    {
                        "id": link,  # the article URL itself — stable and unique, same dedup key as before
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "source": source,
                    }
                )
            except Exception as e:
                log(f"⚠️ Skipped a malformed Zerodha Pulse item: {e}")
                continue

        log(f"Fetched {len(all_entries)} usable entries from Zerodha Pulse.")

    except Exception as e:
        log(f"❌ Failed to parse Zerodha Pulse HTML: {e}")

    return all_entries


# ============================================================================
# 2. AI IMPACT ANALYSIS (Google GenAI SDK — Gemini)
# ============================================================================
ANALYSIS_PROMPT_TEMPLATE = """You are a Principal Equity Research Analyst specializing in Indian stock markets (NSE/BSE).

Analyze the following news item strictly for its TRADEABLE market impact.

RULES:
- IGNORE only genuine noise: routine end-of-day wrap-ups with no new information, pure opinion columns, listicles, generic "markets closed higher/lower" summaries, and repeated coverage of something already fully priced in with nothing new added.
- Flag anything with a plausible, specific effect on a stock, sector, Nifty, or BankNifty — HIGH impact for clear major events (earnings surprises, M&A, regulatory action, management changes, large orders/contracts, guidance changes, macro/policy shocks, rating actions, block deals, litigation, capacity expansion), and MEDIUM impact for smaller but still concrete developments (notable single-stock price moves, sectoral trends, broker upgrades/downgrades, partnership announcements, early-stage regulatory or policy signals, notable promoter/insider activity).
- When genuinely uncertain whether something is significant, prefer flagging it as MEDIUM impact over discarding it — a false positive costs one Telegram message, a false negative costs a missed trade signal.
- If the news is truly low impact or routine noise, respond with EXACTLY the single word: IGNORE
- Do not explain your reasoning. Do not add any text outside the specified format.

If the news IS high or medium impact, respond in EXACTLY this Telegram Markdown format (no extra commentary, no code fences):

🚨 **MARKET IMPACT ALERT** 🚨

📌 **Stock / Entity:** [Exact Company Name / Nifty / BankNifty]
💥 **Impact Level:** [🔴 HIGH IMPACT / 🟠 MEDIUM IMPACT]

📝 **Summary:**
• [First concise, actionable bullet point]
• [Second concise, actionable bullet point]

📊 **Market Bias:** [📈 BULLISH / 📉 BEARISH / ⚖️ NEUTRAL] - [5-word reasoning]

---
NEWS TITLE: {title}
NEWS SUMMARY: {summary}
SOURCE: {source}
---
"""

_genai_client = None


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _is_transient_gemini_error(e: Exception) -> bool:
    """503 UNAVAILABLE, 429 rate limits, and timeouts are temporary —
    worth retrying within seconds. 400/401/403/404 mean something is
    actually wrong (bad key, bad model, bad request) and retrying won't help."""
    text = str(e).upper()
    return any(marker in text for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "TIMEOUT", "DEADLINE"))


def _is_daily_quota_exhausted(e: Exception) -> bool:
    """A RESOURCE_EXHAUSTED tied to a *per-day* quota won't recover within
    a short backoff window — retrying 2s/4s/8s later is pointless. Detected
    from Google's quotaId, e.g. 'GenerateRequestsPerDayPerProjectPerModel'."""
    text = str(e).upper().replace("_", "")
    return "RESOURCEEXHAUSTED" in text and "PERDAY" in text


def analyze_news(entry: dict):
    """Returns:
      - the formatted alert text, if high/medium impact
      - the string "IGNORE" if the model judged it low-impact noise
        (safe to mark processed — we never want to re-analyze it)
      - None if every candidate model failed, or if no Gemini model is
        currently reachable at all, or the daily budget is exhausted
        (must NOT be marked processed — retry it next cycle)

    Retry/fallback strategy: for each candidate model (starting with the
    currently resolved one, then the rest of GEMINI_MODEL_CANDIDATES in
    order), retries up to GEMINI_MAX_RETRIES_PER_MODEL times with
    exponential backoff + jitter on transient errors (503/429/timeouts).
    If a model's retries are exhausted, OR its error indicates a daily
    quota exhaustion (no point retrying that one further), we fall through
    to the next candidate model immediately, within this same call — a
    503 from high demand on one model doesn't mean the next is also down.
    """
    global MODEL_NAME

    if not MODEL_NAME:
        # Gemini isn't connected right now (outage/quota exhaustion). Don't
        # call the API at all — just leave this item for a later cycle.
        return None

    if gemini_budget_remaining() <= 0:
        # Our own daily budget (not just Google's) is used up. Skip without
        # calling the API — leave unprocessed for a later cycle/day.
        return None

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        title=entry["title"],
        summary=entry["summary"] or "N/A",
        source=entry["source"],
    )
    client = get_genai_client()
    last_error = None

    # Try the currently resolved model first, then fall back through the
    # rest of the candidate list, in priority order, skipping duplicates.
    models_to_try = [MODEL_NAME] + [m for m in GEMINI_MODEL_CANDIDATES if m != MODEL_NAME]

    for model_id in models_to_try:
        if gemini_budget_remaining() <= 0:
            break

        for attempt in range(1, GEMINI_MAX_RETRIES_PER_MODEL + 1):
            if gemini_budget_remaining() <= 0:
                log(f"🛑 Daily Gemini call budget exhausted mid-attempt for '{entry['title'][:50]}'. Stopping.")
                return None

            try:
                record_gemini_call()
                response = client.models.generate_content(model=model_id, contents=prompt)
                text = (getattr(response, "text", "") or "").strip()

                if not text:
                    # Empty response with no exception — treat as a soft failure,
                    # worth a retry rather than silently marking it "ignored".
                    last_error = "empty response from model"
                    raise ValueError(last_error)

                if model_id != MODEL_NAME:
                    log(f"🔀 Falling back to model '{model_id}' (issues with '{MODEL_NAME}') — "
                        f"and adopting it as primary going forward.")
                    MODEL_NAME = model_id

                if text.upper() == "IGNORE" or text.upper().startswith("IGNORE"):
                    return "IGNORE"

                return text

            except Exception as e:
                last_error = e

                if _is_daily_quota_exhausted(e):
                    # No point retrying this model further today — move to
                    # the next candidate immediately instead of burning
                    # backoff time on an exhausted quota.
                    log(f"⚠️ Daily quota exhausted for model '{model_id}': {e}")
                    if model_id == MODEL_NAME:
                        MODEL_NAME = None  # forces resolve_gemini_model() to pick a fresh one next cycle
                    break  # stop retrying this model, fall through to next candidate

                if _is_transient_gemini_error(e) and attempt < GEMINI_MAX_RETRIES_PER_MODEL:
                    delay = GEMINI_RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, GEMINI_RETRY_JITTER_SECONDS)
                    log(f"⚠️ Gemini transient error on attempt {attempt}/{GEMINI_MAX_RETRIES_PER_MODEL} "
                        f"for model '{model_id}' on '{entry['title'][:50]}': {e} — retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    # Retries exhausted for this model, or a non-transient
                    # error (e.g. this model is deprecated/404) — fall
                    # through to the next candidate model.
                    log(f"⚠️ Model '{model_id}' failed for '{entry['title'][:50]}' "
                        f"({e}) — trying next fallback candidate.")
                    break

    log(f"❌ All Gemini candidates failed for '{entry['title'][:60]}': {last_error}")
    return None


# ============================================================================
# 3. TELEGRAM ALERT DISPATCH
# ============================================================================
def send_telegram_alert(message: str, link: str) -> bool:
    full_message = f"{message}\n\n🔗 [Read Full Article]({link})"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": full_message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        r = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return True
        log(f"❌ Telegram HTTP {r.status_code}: {r.text[:300]}")
        return False
    except requests.exceptions.Timeout:
        log("❌ Telegram request timed out.")
        return False
    except requests.exceptions.RequestException as e:
        log(f"❌ Telegram request error: {e}")
        return False


# ============================================================================
# MAIN CYCLE + CONTINUOUS LOOP
# ============================================================================
def _per_cycle_gemini_cap() -> int:
    """Spreads the daily budget evenly across the day's scan cycles, so one
    busy hour (e.g. market open) can't burn the whole day's quota and leave
    nothing for the rest of the day — but never drops below
    GEMINI_MIN_ITEMS_PER_CYCLE, so a single failed item can't stall an
    entire cycle the way a cap of 1 did. The daily budget check further
    down is the actual hard ceiling on real API spend, not this number."""
    cycles_per_day = max(1, 86400 // CHECK_INTERVAL_SECONDS)
    even_spread = max(1, GEMINI_DAILY_CALL_BUDGET // cycles_per_day)
    return max(even_spread, GEMINI_MIN_ITEMS_PER_CYCLE)


def run_cycle() -> None:
    log("--- Starting scan cycle ---")

    if not MODEL_NAME:
        if not resolve_gemini_model():
            log("⏳ Gemini still unavailable this cycle (cooldown, outage, or budget exhausted). "
                "Skipping this scan — Telegram bot stays up and will keep retrying automatically.")
            return

    remaining_today = gemini_budget_remaining()
    if remaining_today <= 0:
        log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) already used today. "
            f"Skipping this scan — resumes automatically after midnight UTC.")
        return

    cycle_cap = _per_cycle_gemini_cap()
    log(f"Gemini budget: {remaining_today}/{GEMINI_DAILY_CALL_BUDGET} left today, "
        f"aiming to evaluate up to {cycle_cap} item(s) this cycle.")

    entries = fetch_all_feeds()
    log(f"Total entries fetched: {len(entries)}")

    new_alerts = 0
    items_evaluated = 0  # only decisive outcomes count — failures don't consume a slot
    for entry in entries:
        if is_news_processed(entry["id"]):
            continue

        if items_evaluated >= cycle_cap:
            log(f"⏭️  Per-cycle evaluation target ({cycle_cap}) reached — remaining fresh items "
                f"will be picked up next cycle.")
            break

        if gemini_budget_remaining() <= 0:
            log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) exhausted mid-cycle. "
                f"Stopping — remaining items will be picked up once the budget resets.")
            break

        analysis = analyze_news(entry)

        if analysis is None:
            # Gemini call failed even after its own internal retries, or the
            # daily budget ran out mid-attempt. Do NOT mark as processed —
            # it'll be retried on a later scan — and do NOT count this
            # against the per-cycle cap, so one bad 503 doesn't cost the
            # cycle its only shot at evaluating real news.
            log(f"⏭️  Leaving unprocessed for retry next cycle: {entry['title'][:70]}")
            continue

        items_evaluated += 1

        if analysis == "IGNORE":
            # Genuinely low-impact — safe to mark so we never re-spend an
            # API call analyzing it again.
            mark_news_processed(entry["id"], entry["title"], entry["source"], "ignored")
            continue

        if send_telegram_alert(analysis, entry["link"]):
            mark_news_processed(entry["id"], entry["title"], entry["source"], "sent")
            new_alerts += 1
            log(f"✅ Alert sent: {entry['title'][:70]}")
        else:
            log(f"⚠️ Delivery failed, will retry next cycle: {entry['title'][:70]}")

        time.sleep(1.5)

    log(f"--- Cycle complete: {new_alerts} new alert(s) sent, "
        f"{gemini_calls_used_today()}/{GEMINI_DAILY_CALL_BUDGET} Gemini calls used today ---")


def main() -> None:
    log("=" * 60)
    log(" AlphaRadar AI — Indian Market News Filter starting up ")
    log("=" * 60)

    # Bind the health-check port immediately, before any config validation
    # or network calls — Render's port scan has its own timeout independent
    # of whether Telegram/Gemini are configured correctly, so this must not
    # wait on (or fail because of) those checks.
    start_health_check_server()
    start_self_ping_thread()

    if not validate_config():
        log("🛑 Fix the config values above and re-run the cell. Stopping now.")
        return

    init_db()

    telegram_ok = test_telegram_connection()
    if not telegram_ok:
        log("🛑 Telegram connection failed — nothing works without this. Stopping before entering the loop.")
        return

    # Gemini connectivity is checked but NOT fatal: a rate limit or a
    # temporary outage on Google's side should never take the whole bot
    # down. If it's unavailable now, the loop below starts anyway and
    # retries resolve_gemini_model() each cycle (cooldown-limited so it
    # doesn't hammer an exhausted quota).
    gemini_ok = resolve_gemini_model(force=True)
    if not gemini_ok:
        log("⚠️ Gemini is not reachable right now (see errors above). Starting the loop anyway — "
            "it will keep retrying in the background and resume alerts once a model is available.")

    send_startup_notification()

    log("✅ Startup checks complete. Entering continuous monitoring loop.")
    log("   (Runtime > Interrupt execution to stop.)")

    while True:
        try:
            run_cycle()
        except Exception:
            log("❌ Unhandled error in monitoring cycle:")
            traceback.print_exc()

        log(f"Sleeping {CHECK_INTERVAL_SECONDS}s until next scan...")
        try:
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Stopped by user.")
            break


if __name__ == "__main__":
    main()
