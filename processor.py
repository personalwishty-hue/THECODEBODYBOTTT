import os
import json
import time
import logging
import requests
from datetime import datetime, timezone
from supabase import create_client, Client

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ai_processor")

# ── Environment variables ─────────────────────────────────────────────────────
SUPABASE_URL: str     = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str     = os.environ["SUPABASE_KEY"]
GEMINI_API_KEY: str   = os.environ["GEMINI_API_KEY"]
POLL_INTERVAL: int    = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))

GEMINI_MODEL          = "gemini-1.5-flash"
GEMINI_BASE           = "https://generativelanguage.googleapis.com/v1beta/models"

# ── Supabase client ───────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Gemini helpers ────────────────────────────────────────────────────────────
def gemini_generate(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """Call Gemini generateContent and return the text response."""
    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    body = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def gemini_search(query: str) -> str:
    """Call Gemini with Google Search grounding to get live web data."""
    url = f"{GEMINI_BASE}/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "tools": [{"google_search_retrieval": {}}],
        "generationConfig": {"maxOutputTokens": 4096},
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Step 1: Classify Discord message ─────────────────────────────────────────
CLASSIFY_SYSTEM = """You are a content classifier for a gaming website.
Classify Discord messages into one of three categories.
Respond ONLY with valid JSON — no markdown, no explanation.

For a redeemable game code:
{"type":"code","code":"ACTUAL_CODE","game":"Game Name","description":"What it gives"}

For game news/updates/announcements:
{"type":"news","title":"Short title","summary":"1-2 sentences","game":"Game Name"}

For anything else (chat, questions, memes, spam):
{"type":"irrelevant"}"""

def classify_message(content: str) -> dict:
    raw = gemini_generate(
        prompt=f"Classify this Discord message:\n\n{content}",
        system=CLASSIFY_SYSTEM,
    )
    return parse_json(raw)


# ── Step 2a: Code logic — find or create article, then add code ───────────────
def get_codes_article(game: str) -> dict | None:
    """Return existing codes article for this game, or None."""
    resp = (
        supabase.table("code_articles")
        .select("*")
        .ilike("game", game)
        .eq("is_published", True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def build_full_codes_article(game: str) -> dict:
    """Use Gemini Search to research the game and build a complete codes article."""
    log.info("Building new codes article for: %s", game)

    # Search for current codes and game info
    search_result = gemini_search(
        f"All working {game} codes {datetime.now().strftime('%B %Y')} "
        f"redeem codes list how to redeem guide FAQ"
    )

    # Ask Gemini to structure it into a full article
    structure_prompt = f"""Based on this research about {game} codes:

{search_result}

Write a complete, professional codes article for a gaming website.
Respond ONLY with valid JSON, no markdown.

{{
  "title": "{game} Codes ({datetime.now().strftime('%B %Y')}): All Working Codes",
  "summary": "2-sentence intro for the page",
  "working_codes": [
    {{"code": "CODE1", "reward": "What it gives"}},
    {{"code": "CODE2", "reward": "What it gives"}}
  ],
  "expired_codes": ["OLD1", "OLD2"],
  "how_to_redeem": "Step by step instructions as a single string with \\n between steps",
  "faq": [
    {{"question": "Q1", "answer": "A1"}},
    {{"question": "Q2", "answer": "A2"}}
  ],
  "guide_intro": "2-3 paragraphs about the game and why codes are useful"
}}"""

    raw = gemini_generate(structure_prompt, temperature=0.4)
    article_data = parse_json(raw)
    article_data["game"] = game
    return article_data


def create_codes_article(game: str, first_code: dict) -> int:
    """Create a brand new codes article for a game. Returns the new article id."""
    article_data = build_full_codes_article(game)

    # Add the new code from Discord if not already in the list
    new_code_entry = {
        "code": first_code.get("code", ""),
        "reward": first_code.get("description", "")
    }
    working = article_data.get("working_codes", [])
    if not any(c["code"] == new_code_entry["code"] for c in working):
        working.insert(0, new_code_entry)
    article_data["working_codes"] = working

    resp = supabase.table("code_articles").insert({
        "game": game,
        "title": article_data.get("title", f"{game} Codes"),
        "summary": article_data.get("summary", ""),
        "content": json.dumps(article_data),   # full structured content
        "is_published": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

    article_id = resp.data[0]["id"]
    log.info("Created new codes article id=%d for %s", article_id, game)
    return article_id


def append_code_to_article(article: dict, new_code: dict) -> None:
    """Add a new code to an existing codes article."""
    content = json.loads(article.get("content", "{}"))
    working = content.get("working_codes", [])

    code_val = new_code.get("code", "")
    # Avoid duplicates
    if any(c["code"] == code_val for c in working):
        log.info("Code %s already exists in article, skipping.", code_val)
        return

    working.insert(0, {"code": code_val, "reward": new_code.get("description", "")})
    content["working_codes"] = working

    supabase.table("code_articles").update({
        "content": json.dumps(content),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", article["id"]).execute()

    log.info("Appended code %s to article id=%d", code_val, article["id"])


def handle_code(data: dict, source_message_id: int) -> None:
    game = data.get("game", "Unknown")

    # Always save raw code to game_codes table
    supabase.table("game_codes").insert({
        "code": data.get("code", ""),
        "game": game,
        "description": data.get("description", ""),
        "source_message_id": source_message_id,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "is_active": True,
    }).execute()

    # Find or create the codes article
    existing = get_codes_article(game)
    if existing:
        append_code_to_article(existing, data)
    else:
        create_codes_article(game, data)


# ── Step 2b: News logic — create a new article ────────────────────────────────
def handle_news(data: dict, source_message_id: int) -> None:
    game = data.get("game", "Unknown")
    title = data.get("title", "")
    summary = data.get("summary", "")

    # Use Gemini Search to enrich the news article
    log.info("Researching news article: %s", title)
    search_result = gemini_search(f"{game} {title} gaming news details")

    enrich_prompt = f"""You are writing a professional gaming news article.

Topic: {title}
Game: {game}
Summary: {summary}
Research: {search_result}

Write a complete news article. Respond ONLY with valid JSON, no markdown.

{{
  "title": "Final article title",
  "summary": "2 sentence summary",
  "body": "Full article body, 3-5 paragraphs, written professionally and engagingly"
}}"""

    raw = gemini_generate(enrich_prompt, temperature=0.5)
    article = parse_json(raw)

    supabase.table("news_articles").insert({
        "title": article.get("title", title),
        "summary": article.get("summary", summary),
        "body": article.get("body", ""),
        "game": game,
        "source_message_id": source_message_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "is_published": True,
    }).execute()
    log.info("Created news article: %s", article.get("title"))


# ── Database helpers ──────────────────────────────────────────────────────────
def get_unprocessed_messages() -> list:
    resp = (
        supabase.table("discord_messages")
        .select("*")
        .eq("processed", False)
        .order("timestamp", desc=False)
        .limit(20)
        .execute()
    )
    return resp.data or []


def mark_as_processed(message_id: int, classification: str) -> None:
    supabase.table("discord_messages").update(
        {"processed": True, "classification": classification}
    ).eq("id", message_id).execute()


# ── Main loop ─────────────────────────────────────────────────────────────────
def process_messages() -> None:
    messages = get_unprocessed_messages()
    if not messages:
        log.info("No new messages to process.")
        return

    log.info("Processing %d message(s)...", len(messages))

    for msg in messages:
        content = msg.get("message_content", "").strip()
        msg_id = msg["id"]

        if not content:
            mark_as_processed(msg_id, "irrelevant")
            continue

        try:
            result = classify_message(content)
            classification = result.get("type", "irrelevant")

            if classification == "code":
                handle_code(result, msg_id)
            elif classification == "news":
                handle_news(result, msg_id)
            else:
                log.info("Message %d is irrelevant, skipping.", msg_id)

            mark_as_processed(msg_id, classification)

        except json.JSONDecodeError as e:
            log.error("JSON parse error for message %d: %s", msg_id, e)
            mark_as_processed(msg_id, "error")
        except Exception as e:
            log.error("Error processing message %d: %s", msg_id, e, exc_info=True)


def main() -> None:
    log.info("AI Processor (Gemini) started. Polling every %ds.", POLL_INTERVAL)
    while True:
        try:
            process_messages()
        except Exception as e:
            log.error("Unexpected error in main loop: %s", e, exc_info=True)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
