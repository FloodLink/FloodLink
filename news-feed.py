import os
import tweepy
import feedparser
import re
import openai
import json
import time
import random
from datetime import datetime, timedelta

# =========================================================
#              ENV + CONSTANTS + BOOT GUARDS
# =========================================================

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_SECRET = os.getenv("TWITTER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# --- Updated model definitions (October 2025) ---
OPENAI_MODEL = "gpt-5"                     # Replaces GPT-4
XAI_MODEL = "grok-4-fast-reasoning"        # Replaces Grok-2-1212

# =========================================================
#                        TWITTER
# =========================================================

# Authenticate Twitter API (Using API v2)
bearer_client = tweepy.Client(
    bearer_token=TWITTER_BEARER_TOKEN
)  # For reads (OAuth 2.0 app-only)

# Authenticate Twitter API (Using API v2)
twitter_client = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET
)

# Accounts FloodLink may occasionally reply to (FILL IN REAL IDs)
TARGET_ACCOUNTS = {
    "NWS": "454313925",        # Replace with actual user IDs
    "RedCross": "6519522",    # Replace with actual user IDs
    "NOAA": "16105558",
    "CIA": "2359926157",   # Replace with actual user IDs
    "WMO": "14499829",
    "UN": "14159148",
    "UNDRR": "140959349",
}

# How many recent tweets we READ per reply run
REPLY_FETCH_LIMIT = 5  # 5 minimum enforced by X


# =========================================================
#                     STORAGE + LIMITS
# =========================================================

LOG_FILE = "floodlink_news.json"
REPLY_LOG_FILE = "floodlink_replies.json"
TARGET_TWEETS_LOG = "floodlink_target_engagement.json"
MENTIONS_REPLY_LOG = "floodlink_mentions_reply_log.json"
MENTIONS_RATE_LIMIT_FILE = "floodlink_last_mentions_check.txt"

RETENTION_DAYS = 10

# Scoring thresholds
NEWS_MIN_SCORE = 9
REPLY_MIN_SCORE = 2
QUOTE_MIN_SCORE = 5
REPOST_MIN_SCORE = 4
LIKE_MIN_SCORE = 4

# Tweet type probabilities
RANDOM_NEWS = 0.3
RANDOM_STATISTIC = 0.2
RANDOM_INFRASTRUCTURE = 0.1
RANDOM_REPLY = 0.15
RANDOM_ENGAGEMENT = 0.15
RANDOM_NONE = 0.1

# Engagement weights
ENGAGEMENT_QUOTE_WEIGHT = 0.5
ENGAGEMENT_REPOST_WEIGHT = 0.5
ENGAGEMENT_LIKE_WEIGHT = 0.0

# Daily tweet limits
NEWS_TWEETS_LIMIT = 4        
STAT_TWEETS_LIMIT = 2
INFRA_TWEETS_LIMIT = 1
REPLY_TWEETS_LIMIT = 0
MENTIONS_REPLY_DAILY_LIMIT = 6

# Daily limits for retweets/quotes (adjust as needed)
DAILY_QUOTE_LIMIT = 1
DAILY_REPOST_LIMIT = 2
DAILY_LIKE_LIMIT = 0   # Very safe

# =========================================================
#                         RSS
# =========================================================

# FloodLink-focused Google News RSS feeds
RSS_FEEDS = [
    # Global flood & flash-flood alerts
    "https://news.google.com/rss/search?q=flood+warning+OR+flash+flood&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=heavy+rain+flooding+OR+river+overflows&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=dam+break+flood+OR+levee+breach&hl=en&gl=US&ceid=US:en",

    # Tropical cyclones with flooding
    "https://news.google.com/rss/search?q=hurricane+flooding+OR+storm+surge+flooding&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=typhoon+flooding+OR+cyclone+flooding&hl=en&gl=US&ceid=US:en",

    # Seasonal / monsoon + landslides from heavy rain
    "https://news.google.com/rss/search?q=monsoon+floods+OR+monsoon+flooding&hl=en&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=landslide+heavy+rain+OR+rainfall+triggered+landslide&hl=en&gl=US&ceid=US:en",

    # (Optional later) Global disaster feeds like GDACS / FloodList
    # "https://www.gdacs.org/xml/rss.xml",
]


# =========================================================
#                        HELPERS
# =========================================================

STOPWORDS = set([
    "the", "and", "is", "in", "on", "at", "to", "of", "for", "with", "a", "an",
    "this", "that", "from", "by", "as", "it", "its", "was", "were", "are", "be",
    "new", "latest", "after", "before", "during", "amid"
])

def extract_key_terms(text):
    if not text:
        return set()
    text = str(text).lower()
    words = re.findall(r"\b\w+\b", text)
    numbers = re.findall(r"\d+", text)
    keywords = [w for w in words if w not in STOPWORDS] + numbers
    return set(keywords)

def is_similar_news(new_title, new_summary, processed_articles, threshold=0.6, limit=30):
    new_keywords = extract_key_terms(new_title) | extract_key_terms(new_summary)

    # keep only valid, high-score recent ones
    recent_articles = [
        a for a in processed_articles
        if isinstance(a.get("score", 0), (int, float))
        and a.get("score", 0) >= NEWS_MIN_SCORE
    ][-limit:]

    for article in recent_articles:
        old_keywords = (
            extract_key_terms(article.get("tweet", "")) |
            extract_key_terms(article.get("title", "")) |
            extract_key_terms(article.get("summary", ""))
        )
        if old_keywords:
            similarity = len(new_keywords & old_keywords) / len(new_keywords | old_keywords)
            if similarity >= threshold:
                print(f"⚠️ Skipping similar news: {new_title} (Similarity: {similarity:.2f})")
                return True
    return False

def load_processed_articles():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
            valid = [a for a in data if isinstance(a, dict) and "date" in a]
            print(f"Loaded {len(valid)} processed flood articles.")
            return valid
        except json.JSONDecodeError:
            print("⚠️ Corrupted floodlink_news.json, resetting.")
            return []
    return []

def cleanup_old_articles(processed_articles):
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    return [
        a for a in processed_articles
        if datetime.strptime(a["date"], "%Y-%m-%d") >= cutoff
    ]

def save_processed_articles(processed):
    print("💾 Writing to floodlink_news.json...")
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(processed, f, indent=4)
        print("✅ Successfully wrote to floodlink_news.json!")
    except Exception as e:
        print(f"❌ Error writing to JSON: {e}")
        return

    if os.getenv("GITHUB_ACTIONS"):
        print("🔄 Committing changes to GitHub...")
        os.system("git config --global user.email 'github-actions@github.com'")
        os.system("git config --global user.name 'GitHub Actions'")
        os.system("git add floodlink_news.json")
        commit_result = os.system("git commit -m 'Update floodlink_news.json [Automated]'")
        if commit_result != 0:
            print("⚠️ No changes to commit. Skipping push.")
            return
        push_result = os.system("git push origin main")
        if push_result != 0:
            print("❌ Push failed, check GitHub Actions permissions.")
        else:
            print("✅ Changes committed to GitHub.")

# HELPER FOR QUOTE REPOST / REPOST / LIKES
def load_target_tweets():
    if os.path.exists(TARGET_TWEETS_LOG):
        try:
            with open(TARGET_TWEETS_LOG, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_target_tweets(data):
    with open(TARGET_TWEETS_LOG, "w") as f:
        json.dump(data, f, indent=4)

def cleanup_target_tweets():
    data = load_target_tweets()
    cutoff = (datetime.utcnow() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    cleaned = {tid: entry for tid, entry in data.items() if entry.get("date", "0000-00-00") >= cutoff}
    save_target_tweets(cleaned)
    return cleaned

# HELPER FOR REPLY MENTION
def load_mentions_reply_log():
    if os.path.exists(MENTIONS_REPLY_LOG):
        try:
            with open(MENTIONS_REPLY_LOG, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_mentions_reply_log(data):
    try:
        with open(MENTIONS_REPLY_LOG, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {MENTIONS_REPLY_LOG}")
    except Exception as e:
        print(f"Failed to save {MENTIONS_REPLY_LOG}: {e}")

# Check if we can fetch mentions (Free tier: 1 request every 15 minutes)
def can_check_mentions():
    if not os.path.exists(MENTIONS_RATE_LIMIT_FILE):
        print("DEBUG: No rate limit file exists, allowing check.")
        return True
    try:
        last_check = float(open(MENTIONS_RATE_LIMIT_FILE).read().strip())
        time_since = time.time() - last_check
        print(f"DEBUG: Last check {time_since:.0f} seconds ago.")
        return time_since >= 960  # Increased to 16 min for safety
    except Exception as e:
        print(f"DEBUG: Error reading rate limit file: {e}. Allowing check.")
        return True

def update_mentions_timestamp():
    try:
        with open(MENTIONS_RATE_LIMIT_FILE, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        print(f"Failed to update mentions timestamp: {e}")

def select_tweet_type():
    return random.choices(
        ["news", "statistical", "infrastructure", "reply", "engagement", "none"],
        [RANDOM_NEWS, RANDOM_STATISTIC, RANDOM_INFRASTRUCTURE, RANDOM_REPLY, RANDOM_ENGAGEMENT, RANDOM_NONE]
    )[0]

def count_news_tweets_today(processed_articles):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for a in processed_articles if a.get("date") == today and a.get("type") == "news")

def count_stat_tweets_today(processed_articles):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for a in processed_articles if a.get("date") == today and a.get("type") == "statistical")

def count_infra_tweets_today(processed_articles):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for a in processed_articles if a.get("date") == today and a.get("type") == "infrastructure")

# Count how many crypto tweets were posted today.
def count_engagement_action(data, action):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in data.values() if entry.get("date") == today and entry.get("action") == action)

# Count how many real @-mention replies we made today
def count_mentions_replies_today(log):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in log.values() if entry.get("date") == today)


# =========================================================
#                   NEWS FETCH + SCORING
# =========================================================

def get_latest_news():
    """
    Fetch recent flood-related stories from RSS feeds.
    By default we accept items from the last 6 hours (tune as needed).
    """
    news_list = []
    now = datetime.utcnow()

    for feed_url in RSS_FEEDS:
        try:
            print(f"🔄 Fetching news from: {feed_url}")
            feed = feedparser.parse(feed_url)
            if not feed.entries:
                print(f"⚠️ No entries for {feed_url}")
                continue

            for entry in feed.entries:
                title = entry.title
                link = entry.link
                published_time = datetime(*entry.published_parsed[:6]) if "published_parsed" in entry else now
                source = getattr(entry, "source", None).title if hasattr(entry, "source") else "Unknown source"
                summary = getattr(entry, "summary", "") or ""

                if now - published_time < timedelta(hours=6):
                    news_list.append((title, link, source, summary))
        except Exception as e:
            print(f"❌ Error fetching feed {feed_url}: {e}")
            continue

    return news_list

# =========================================================
#               AI: SCORING + SUMMARIZATION
# =========================================================

def get_news_relevance_score(title, summary):
    """
    Score how relevant this article is to FloodLink (0–10).
    High scores = strong, clear flood / flash-flood signal and impact.
    """
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    prompt = f"""
You are ranking news articles for FloodLink, a global flood-risk early warning system on X.

Assign a relevance score from 0 to 10 for this article, focusing ONLY on:
- river floods
- flash floods
- coastal flooding / storm surge
- flooding from tropical cyclones, typhoons, monsoon rains
- rainfall-triggered landslides
- serious flood preparedness, evacuations, warnings, or post-event impact.

Scoring:
- 9–10: Major or severe floods or flash floods; large areas or populations affected; deaths, missing people, evacuations, red alerts, dam breaks, levee failures, or official high-level flood warnings.
- 7–8: Strong flood risk or heavy rainfall with credible probability of flooding; regional alerts; serious infrastructure damage or clear risk escalation.
- 5–6: Local floods with limited impact, or early signals where the flood angle is present but not yet severe.
- 1–4: Weather stories with weak or indirect flood relevance (e.g., storms but no flooding, vague references, minor local incidents).
- 0: NOT relevant to FloodLink (e.g., generic climate politics, non-weather news, economic climate, sports, entertainment).

Reply with ONLY a single integer (0–10).

Title: {title}
Summary: {summary}
"""

    try:
        response = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        score_text = response.choices[0].message.content.strip()
        score = int(score_text)
        return score if 0 <= score <= 10 else 0
    except Exception as e:
        print(f"❌ Error scoring news: {e}")
        return 0

def summarize_news(title, summary, source):
    """
    Create a FloodLink tweet with clear FORECAST / POST-EVENT label.
    """
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    prompt = f"""
You post as FloodLink, a global flood-risk early warning system on X.

Write ONE tweet about this article in EXACTLY this format:

<FLAG(optional)> <STATUS> <Location>: <short description>

Rules:
- STATUS = FORECAST (future or imminent risk, warnings, alerts)
          OR POST-EVENT (flood already happened: damage, deaths, rescues).
- If you can infer a clear country, put its flag emoji first (e.g. 🇺🇸). Otherwise omit the flag.
- Location: short city/region/country name.
- Description: mention type (flood / flash flood / storm surge / landslide from rain)
  and key risk/impact. If you mention a source, keep it short at the end.
- Max 260 characters total.
- NO hashtags, NO emojis except the optional country flag.
- NO quotation marks.

Title: {title}
Summary: {summary}
Source: {source}
"""

    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    tweet = response.choices[0].message.content.strip()
    tweet = tweet.replace('"', "").replace("'", "")
    return tweet[:280]

def geocode_news_location(title, summary):
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    prompt = f"""
You are a geocoding assistant for FloodLink, a global flood-risk early warning system.

From the news article below, infer the SINGLE most relevant physical location
where the flooding (or flood risk) is happening.

Return a JSON object with EXACTLY these keys:

- "country": Country name (e.g. "Brazil", "India", "United States"), or null if unknown.
- "name": Short human-readable place name (city/town or region, optionally including state), e.g. "Porto Alegre", "Rio Grande do Sul", "Southern Luzon", or "Queensland".
- "latitude": Decimal degrees WGS84 (float, north positive, south negative).
- "longitude": Decimal degrees WGS84 (float, east positive, west negative).
- "confidence": Float between 0 and 1 indicating how sure you are.

If you truly cannot infer any location, return:

{{
  "country": null,
  "name": null,
  "latitude": null,
  "longitude": null,
  "confidence": 0.0
}}

IMPORTANT:
- Answer with ONLY a JSON object, no commentary.
- Prefer the *most specific* place where impacts occur (city/town > region > country).

Title: {title}
Summary: {summary}
"""

    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()

        data = json.loads(raw)

        if not isinstance(data, dict):
            return None

        # Normalise keys & types
        country = data.get("country")
        name = data.get("name")
        lat = data.get("latitude")
        lon = data.get("longitude")
        conf = data.get("confidence", 0.0)

        # Cast lat/lon
        try:
            lat = float(lat) if lat is not None else None
        except (ValueError, TypeError):
            lat = None
        try:
            lon = float(lon) if lon is not None else None
        except (ValueError, TypeError):
            lon = None

        # Cast confidence and clamp 0–1
        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        return {
            "country": country,
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "confidence": conf
        }

    except Exception as e:
        print(f"❌ Error geocoding location: {e}")
        return None


# =========================================================
#      AI: FLOOD STATISTICAL TWEETS
# =========================================================

STATISTICAL_CATEGORIES = [
    "global flood fatalities and trends",
    "population living in floodplains",
    "urban areas exposed to river flooding",
    "coastal cities at risk from sea-level rise and storm surge",
    "economic losses from floods and flash floods",
    "extreme rainfall trends in major cities",
    "monsoon flood patterns in Asia",
    "pluvial (surface) flooding in dense cities",
    "coverage of flood early warning systems worldwide",
    "dams, levees and reservoirs used for flood control"
]

def generate_statistical_tweet(selected_category):
    """
    Generate a global/regional flood statistic tweet.
    """
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    tweet_formats = {
        1: "A single striking statistic or future projection.",
        2: "A direct comparison between two regions or time periods.",
        3: """A short ranked list (3–5 items) under 280 characters.

Format:
<Very short overview of the metric description>

1. Item
2. Item
3. Item
4. Item
5. Item
"""
    }

    selected_format_key = random.choice(list(tweet_formats.keys()))
    selected_format = tweet_formats[selected_format_key]

    prompt = f"""
Assume the current year is 2025.

Generate a concise, factual tweet about **{selected_category}**,
focusing ONLY on floods, flash floods, storm surge, or extreme rainfall.

{selected_format}

Rules:
- Use recent data (2020 onwards) or realistic near-future projections.
- Present only clear numbers or rankings (people, % exposed, losses, etc.).
- Max 280 characters.
- NO hashtags, NO emojis except country flags before location names.
- Use line breaks only if they improve readability.
"""

    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()[:280]

# =========================================================
#      AI: FLOOD INFRASTRUCTURE TWEETS
# =========================================================

def generate_infrastructure_tweet():
    """
    Generate a tweet about physical or digital infrastructure
    related to flood risk: levees, storm tanks, pumps, sensors, etc.
    """
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    prompt = """
Assume the current year is 2025.

Write a concise tweet about infrastructure that protects
or is exposed to floods (levees, dikes, dams, stormwater tanks,
drainage networks, pumping stations, retention basins,
flood sensors or early warning systems).

Rules:
- Focus on ONE clear quantitative metric
  (e.g. km of levees, storage volume, people protected,
   % of city covered by sensors, number of storm tanks, etc.).
- You may highlight a specific country or city if helpful.
- Max 280 characters.
- NO hashtags, NO emojis except country flags before location names.
- Avoid generic marketing language; keep it data-driven.
"""

    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()[:280]

# =========================================================
#                       REPLIES
# =========================================================

def load_reply_log():
    """ Load previously replied tweets to avoid duplicates. """
    if os.path.exists(REPLY_LOG_FILE):
        with open(REPLY_LOG_FILE, "r") as file:
            return json.load(file)
    return {}

def save_reply_log(log_data):
    """ Save replied tweets to prevent duplicate replies. """
    print("💾 Writing to replied_tweets.json...")
    try:
        with open(REPLY_LOG_FILE, "w") as file:
            json.dump(log_data, file, indent=4)
        print("✅ Successfully wrote to replied_tweets.json!")
    except Exception as e:
        print(f"❌ Error writing to replied_tweets.json: {e}")

def count_replies_today(reply_log):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in reply_log.values() if entry["date"] == today)

def fetch_latest_tweets(user_id, max_results=REPLY_FETCH_LIMIT):
    try:
        tweets = bearer_client.get_users_tweets(
            id=user_id,
            max_results=max_results,
            tweet_fields=["text", "created_at"],
            exclude=["retweets", "replies"]
        )
        if not tweets.data:
            return []

        log = load_target_tweets()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        saved = 0

        for tweet in tweets.data:
            tid = str(tweet.id)
            if tid in log:
                continue  # already have it

            # Score relevance immediately
            score = classify_mention_relevance(tweet.text)
            handle = next((k for k, v in TARGET_ACCOUNTS.items() if v == user_id), "unknown")

            log[tid] = {
                "tweet_id": tid,
                "text": tweet.text,
                "author_id": user_id,
                "author_handle": handle,
                "date": today,
                "relevance_score": score,   # ← now 0–10 integer
                "action": None
            }
            saved += 1

        if saved > 0:
            save_target_tweets(log)
            print(f"Saved {saved} new tweets with relevance scores")

        return tweets.data

    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

def generate_grok_reply(tweet_text, username):
    """ Use Grok-2-1212 to generate a smart, relevant reply based on the tweet. """
    prompt = f"""
    You are responding to @{username} on Twitter.

    - Read the following tweet and generate a **concise, data-driven reply** that adds a relevant statistic or fact. 
    - If relevant, bring it towards the flooding theme, but do not force it.
    - Ensure the response is **engaging, contextually relevant, and under 280 characters.**
    - The reply should **enhance the conversation** by providing a valuable insight related to the tweet's topic.
    - **Maintain a professional yet conversational tone.**
    - **DO NOT mention or @ the username to keep it natural.**
    - **DO NOT use hashtags, emojis, or generic phrases.**
    - If no suitable statistic is available, provide a **thoughtful industry insight, preferably related to one of @{username}'s companies.**

    **Tweet:** "{tweet_text}"

    Reply directly with only the final tweet text, nothing else:
    """

    client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()
    
def reply_to_random_tweet():
    """ Randomly select a user, fetch their latest tweet, and reply once per tweet. """
    reply_log = load_reply_log()

    # Daily limit check (using already loaded log)
    if count_replies_today(reply_log) >= REPLY_TWEETS_LIMIT:
        print(f"🚫 Reached daily reply limit ({REPLY_TWEETS_LIMIT}). Exiting.")
        return

    if not TARGET_ACCOUNTS:
        print("⚠️ No TARGET_ACCOUNTS configured. Skipping reply run.")
        return

    # **Step 1: Randomly choose a user**
    user_to_fetch = random.choice(list(TARGET_ACCOUNTS.keys()))
    user_id = TARGET_ACCOUNTS[user_to_fetch]
    print(f"🔍 Fetching tweets from @{user_to_fetch}...")

    # **Step 2: Fetch their latest tweets**
    all_tweets = fetch_latest_tweets(user_id)  # uses REPLY_FETCH_LIMIT

    if not all_tweets:
        print(f"🔍 No tweets found for @{user_to_fetch}.")
        return

    # Build set of tweet IDs we've already replied to
    replied_ids = set(reply_log.keys())

    # **Step 3: Filter out tweets we've already replied to**
    new_tweets = [t for t in all_tweets if str(t.id) not in replied_ids]

    if not new_tweets:
        print(f"🔁 All recent tweets from @{user_to_fetch} already replied to. Skipping this run.")
        return

    # # SMART FILTER: with a score over defined and pick the most recent new tweet
    selected_tweet = new_tweets[0]
    tweet_id = selected_tweet.id                  # ← fixed
    tweet_text = selected_tweet.text              # ← fixed

    target_data = load_target_tweets()
    score = target_data.get(str(tweet_id), {}).get("relevance_score", 0)

    if score <= REPLY_MIN_SCORE:
        print(f"Skipping reply → low relevance score {score}/10: \"{selected_tweet.text[:80]}...\"")
        return
        
    username = user_to_fetch  # Using stored username

    # **Step 4: Generate a Grok-powered reply**
    reply_text = generate_grok_reply(tweet_text, username)
    if not reply_text:
        print(f"❌ Failed to generate reply for @{username}. Skipping.")
        return

    # **Step 5: Post the reply**
    try:
        twitter_client.create_tweet(
            text=reply_text,  # No @{username} prefix to keep it natural
            in_reply_to_tweet_id=tweet_id
        )
        print(f"✅ Replied to @{username}: {reply_text}")

        # **Step 6: Log replied tweet (now with full texts)**
        reply_log[str(tweet_id)] = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "username": username,
            "tweet_id": tweet_id,
            "source_text": tweet_text,
            "reply_text": reply_text,
            "relevance_score": score
        }
        save_reply_log(reply_log)

    except tweepy.errors.TweepyException as e:
        print(f"❌ Error posting reply: {e}")

# =========================================================
#             TARGET ENGAGEMENT (Quote/RT/Like)
# =========================================================

def classify_mention_relevance(text):
    client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    
    prompt = f"""
    Score 0–10 (integer only) the relevance of this tweet for FloodLink (@FloodLink), a global early-warning network focused exclusively on flooding risks, flood emergencies, and flood-protection infrastructure.
    
    10 = Active catastrophic flooding, flash floods, dam/levee failure, mass evacuations, deaths, red alerts in progress
    9  = Major ongoing or imminent severe flooding (large populations/regions at risk, official emergency declarations)
    8  = Credible high-risk flood warnings, storm surge threats, extreme rainfall + river overflow forecasts
    7  = Heavy rainfall events very likely to cause serious flooding, urban pluvial flooding alerts, cyclone/monsoon flood risk
    6  = Moderate/local flooding already occurring or forecasted, landslide risk from rain, coastal flood advisories
    5  = General extreme rainfall, tropical cyclones, or weather systems that could evolve into flooding
    4  = Flood-adjacent topics (dams, levees, drainage upgrades, early-warning systems, nature-based solutions)
    3  = Climate/disaster-resilience discussion, light weather memes from trusted accounts
    2  = Off-topic but not spam (e.g., general weather, earthquakes, wildfires)
    0–1 = gm/gn, pure spam, crypto promos, unrelated politics, one-word replies
    
    Only reply with a single integer 0–10. No explanation.
    
    Tweet: \"{text}\"
    """
    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.1
        )
        score_text = resp.choices[0].message.content.strip()
        score = int(score_text)
        return max(0, min(10, score))  # Clamp to 0–10
    except:
        return 0

def generate_quote_comment(text):
    client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    
    prompt = f"""
    Write a sharp, professional quote tweet (max 180 chars) that adds a precise insight or data point to the original tweet.
    
    Rules:
    - Generate a **concise, data-driven insight** that adds a relevant statistic or fact. 
    - No hashtags, no @-mentions, no generic emojis (flags OK)
    - Sound forward-looking and authoritative
    - Never generic — always add a concrete angle or number when possible
    
    Original tweet: {text}
    
    Quote comment only:"""

    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=110,
            temperature=0.8
        )
        comment = resp.choices[0].message.content.strip()
        return comment[:180] if comment else None
    except:
        return None

def process_mention_engagement():
    data = cleanup_target_tweets()  # Auto-remove old tweets
    if not data:
        print("No target tweets in pool")
        return

    # Filter tweets that haven't been engaged with yet
    available = [(tid, entry) for tid, entry in data.items() if entry.get("action") is None]
    if not available:
        print("All target tweets already engaged with")
        return

    # Random-within-random: pick action type
    action = random.choices(
        ["quote", "repost", "like"],
        weights=[ENGAGEMENT_QUOTE_WEIGHT, ENGAGEMENT_REPOST_WEIGHT, ENGAGEMENT_LIKE_WEIGHT],
        k=1
    )[0]

    print(f"Engagement mode: {action.upper()} → curating from target accounts")

    processed = 0
    random.shuffle(available)

    for tid, entry in available:
        text = entry["text"]
        score = entry.get("relevance_score", 0)  # Use pre-scored value

        # QUOTE:
        if (action == "quote" and score >= QUOTE_MIN_SCORE and 
            count_engagement_action(data, "quote") < DAILY_QUOTE_LIMIT):
            comment = generate_quote_comment(text)
            if comment and 15 < len(comment) < 200:
                try:
                    twitter_client.create_tweet(text=comment, quote_tweet_id=int(tid))
                    
                    # ←←← NOW SAVES THE ACTUAL QUOTE TEXT!
                    data[tid]["action"] = "quote"
                    data[tid]["date"] = datetime.utcnow().strftime("%Y-%m-%d")
                    data[tid]["quote_text"] = comment.strip()   # ← THIS IS THE FIX!
                    
                    print(f"Quote-tweeted: {comment[:60]}...")
                    processed += 1
                    save_target_tweets(data)
                    return  # One quote per run is enough
                except Exception as e:
                    print(f"Quote failed: {e}")

        # REPOST: 7–10
        elif (action == "repost" and score >= REPOST_MIN_SCORE and 
              count_engagement_action(data, "repost") < DAILY_REPOST_LIMIT):
            try:
                twitter_client.retweet(tweet_id=int(tid))  # ← This always works
                data[tid]["action"] = "repost"
                data[tid]["date"] = datetime.utcnow().strftime("%Y-%m-%d")
                print("Reposted from target account")
                processed += 1
                if processed >= 2:
                    save_target_tweets(data)
                    return
            except Exception as e:
                print(f"Repost failed: {e}")

        # LIKE: 5–10 (or everything if like mode)
        elif (action == "like" and score >= LIKE_MIN_SCORE and 
              count_engagement_action(data, "like") < DAILY_LIKE_LIMIT):
            try:
                twitter_client.like(tweet_id=int(tid))
                data[tid]["action"] = "like"
                data[tid]["date"] = datetime.utcnow().strftime("%Y-%m-%d")
                processed += 1
            except Exception as e:
                print(f"Like failed: {e}")

        if processed >= 3:
            break

    save_target_tweets(data)
    print(f"Engagement complete: {processed} actions")
    

# =========================================================
#         REAL @-MENTION → ALWAYS REPLY (Separate & Guaranteed)
# =========================================================

MY_USER_ID = None

def get_my_user_id():
    global MY_USER_ID
    if MY_USER_ID:
        return MY_USER_ID
    try:
        MY_USER_ID = twitter_client.get_me().data.id
        print(f"My user ID: {MY_USER_ID}")
        return MY_USER_ID
    except:
        return None

def process_mention_replies():
    if not can_check_mentions():
        print("Mentions check skipped (15-min rate limit)")
        return

    user_id = get_my_user_id()
    if not user_id:
        return

    log = load_mentions_reply_log()
    since_id = log.get('metadata', {}).get('last_mention_id')

    update_mentions_timestamp()  # Commit before fetch

    try:
        resp = bearer_client.get_users_mentions(
            id=user_id,
            max_results=10,
            tweet_fields=["author_id", "text"],
            since_id=since_id
        )
        print(f"Fetched {len(resp.data or [])} mentions")
    except tweepy.errors.TooManyRequests as e:
        print(f"Rate limit hit (429): {e}. Waiting longer next time.")
        return
    except Exception as e:
        print(f"Failed to fetch mentions: {e}")
        return

    mentions = resp.data or []
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Update metadata with max ID from this fetch (even if no replies)
    if mentions:
        new_max = max(int(tweet.id) for tweet in mentions)
        current_max = since_id or 0
        updated_max = max(current_max, new_max)
        if 'metadata' not in log:
            log['metadata'] = {}
        log['metadata']['last_mention_id'] = updated_max
        save_mentions_reply_log(log)  # Save updated max early

    if count_mentions_replies_today(log) >= MENTIONS_REPLY_DAILY_LIMIT:  # ← now uses the correct one
        print(f"Daily mention reply limit reached ({MENTIONS_REPLY_DAILY_LIMIT})")
        return

    replied = 0
    for tweet in mentions:
        tid = str(tweet.id)
        if tid in log or tweet.author_id == user_id:
            continue

        # Blocks: crypto spam (50+ tags) + Grok/Claude/Gemini replies (2 tags) + any mass-tag nonsense
        if len(re.findall(r'@\w+', tweet.text)) > 1:
            print(f"Blocked mention with multiple @ tags ({tweet.text[:100]}...)")
            continue

        reply_text = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1").chat.completions.create(
            model=XAI_MODEL,
            messages=[{
                "role": "user",
                "content": f"""
        You are @FloodLink — an early flood alert network and flood risk infrastructure account.
        
        Someone just @-mentioned you with this:
        
        "{tweet.text}"
        
        Write a concise, natural, professional reply (max 240 chars).
        - No hashtags, no @-mentions (X adds them automatically)
        - No generic emojis (country flags OK)
        - Sound helpful and slightly forward-looking
        
        Reply directly with only the final reply text, nothing else:
        """
            }],
            temperature=0.7,
            max_tokens=300
        ).choices[0].message.content.strip()

        if not reply_text or len(reply_text) > 280:
            continue

        try:
            twitter_client.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet.id)
            log[tid] = {"date": today, "replied": True, "text": reply_text}
            save_mentions_reply_log(log)
            print(f"Replied to @-mention: {reply_text[:60]}...")
            if count_mentions_replies_today(log) >= MENTIONS_REPLY_DAILY_LIMIT:
                break
        except Exception as e:
            print(f"Mention reply failed: {e}")

    if replied:
        print(f"Completed {replied} mention replies")

# =========================================================
#                      POSTING
# =========================================================

def post_tweet(tweet):
    print(f"🚀 Attempting to tweet: {tweet}")
    try:
        resp = twitter_client.create_tweet(text=tweet)
        print(f"✅ Tweet posted: {resp.data}")

        tweet_id = None
        try:
            tweet_id = resp.data.get("id")
        except Exception:
            pass

        # small cooldown so runs don't spam
        time.sleep(120)

        return tweet_id  # None if parsing failed
    except tweepy.errors.Forbidden as e:
        if "Status is a duplicate" in str(e):
            print("⚠️ Duplicate tweet detected. Skipping.")
        else:
            print(f"❌ Twitter API error: {e}")
        return None
    except tweepy.errors.TweepyException as e:
        print(f"❌ Other Tweepy error: {e}")
        return None


# =========================================================
#                        MAIN
# =========================================================

if __name__ == "__main__":
    print("Agent started — checking real @-mentions first...")
    process_mention_replies()
    
    print("🔍 Loading previously processed FloodLink items...")
    processed_articles = load_processed_articles()
    filtered_links = {a.get("link") for a in processed_articles if a.get("link")} if processed_articles else set()
    print(f"📂 {len(processed_articles)} items already processed.")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_news_count = count_news_tweets_today(processed_articles)
    today_stat_count = count_stat_tweets_today(processed_articles)
    today_infra_count = count_infra_tweets_today(processed_articles)
    reply_log = load_reply_log()
    today_reply_count = count_replies_today(reply_log)

    tweet_type = select_tweet_type()
    print(f"🔀 Selected tweet type: {tweet_type}")

    # enforce per-type limits early
    if tweet_type == "news" and today_news_count >= NEWS_TWEETS_LIMIT:
        print(f"🚫 Reached daily news limit ({NEWS_TWEETS_LIMIT}).")
        exit(0)
    if tweet_type == "statistical" and today_stat_count >= STAT_TWEETS_LIMIT:
        print(f"🚫 Reached daily statistical limit ({STAT_TWEETS_LIMIT}).")
        exit(0)
    if tweet_type == "infrastructure" and today_infra_count >= INFRA_TWEETS_LIMIT:
        print(f"🚫 Reached daily infrastructure limit ({INFRA_TWEETS_LIMIT}).")
        exit(0)
    if tweet_type == "reply" and today_reply_count >= REPLY_TWEETS_LIMIT:
        print(f"🚫 Reached daily reply limit ({REPLY_TWEETS_LIMIT}).")
        exit(0)

    # ---------- REPLY ----------
    if tweet_type == "reply":
        reply_to_random_tweet()
        exit(0)

    # ---------- FLOOD NEWS ----------
    if tweet_type == "news":
        latest_news = get_latest_news()
        print(f"📰 Found {len(latest_news)} recent articles.")

        scored_news = []
        seen_links = set()

        for title, link, source, summary in latest_news:
            if today_news_count >= NEWS_TWEETS_LIMIT:
                print(f"🚫 Stopping news: {today_news_count} tweets reached.")
                break

            if link in seen_links or link in filtered_links:
                print(f"⏩ Skipping duplicate article: {title}")
                continue
            seen_links.add(link)

            # similarity filter
            if is_similar_news(title, summary, processed_articles, threshold=0.5, limit=30):
                processed_articles.append({
                    "link": link,
                    "date": today,
                    "title": title,
                    "summary": summary,
                    "similarity_excluded": "Yes",
                    "score": 0,
                    "status": "skipped",
                    "tweet": None
                })
                continue

            score = get_news_relevance_score(title, summary)

            base_entry = {
                "link": link,
                "date": today,
                "title": title,
                "summary": summary,
                "similarity_excluded": "No",
                "score": score,
                "status": "processed",
                "tweet": None
            }
            processed_articles.append(base_entry)
            scored_news.append((score, title, link, source, summary))

        # sort by score
        scored_news.sort(reverse=True, key=lambda x: x[0])

        # respect remaining slots + per-run cap (3)
        remaining_slots = max(0, NEWS_TWEETS_LIMIT - today_news_count)
        if remaining_slots <= 0:
            top_articles = []
        else:
            per_run_cap = 3
            max_to_tweet = min(remaining_slots, per_run_cap)
            top_articles = scored_news[:max_to_tweet]

        for score, title, link, source, summary in top_articles:
            if today_news_count >= NEWS_TWEETS_LIMIT:
                break
            if score >= NEWS_MIN_SCORE:
                tweet = summarize_news(title, summary, source)
                
                # 🔎 NEW: geocode the news location in your desired format
                geo_data = geocode_news_location(title, summary)
                if geo_data:
                    print(
                        f"📍 Geocoded news → {geo_data.get('name')} "
                        f"({geo_data.get('latitude')}, {geo_data.get('longitude')}) "
                        f"country={geo_data.get('country')} "
                        f"conf={geo_data.get('confidence')}"
                    )
        
                tweet_id = post_tweet(tweet)
                if tweet_id:
                    today_news_count += 1
        
                    entry = {
                        "link": link,
                        "date": today,
                        "title": title,
                        "summary": summary,
                        "similarity_excluded": "No",
                        "score": score,
                        "status": "posted",
                        "tweet": tweet,
                        "type": "news",
                        "tweet_id": tweet_id
                    }
        
                    # Attach `geo` in the format you defined
                    if geo_data:
                        entry["geo"] = geo_data
        
                    processed_articles.append(entry)
        
                    # OPTIONAL: back-fill the earlier "processed" entry for this link
                    for a in processed_articles:
                        if a.get("link") == link and a.get("status") == "processed":
                            a["tweet_id"] = tweet_id
                            if geo_data:
                                a["geo"] = geo_data
                            break
        
            else:
                print(f"🚫 Article below threshold (score={score}): {title}")

    # ---------- FLOOD STATISTICS ----------
    elif tweet_type == "statistical":
        if today_stat_count >= STAT_TWEETS_LIMIT:
            print(f"🚫 Reached daily statistical limit ({STAT_TWEETS_LIMIT}).")
        else:
            selected_category = random.choice(STATISTICAL_CATEGORIES)
            tweet = generate_statistical_tweet(selected_category)
            tweet_id = post_tweet(tweet)
            if tweet_id:
                today_stat_count += 1
                processed_articles.append({
                    "link": None,
                    "date": today,
                    "status": "posted",
                    "tweet": tweet,
                    "type": "statistical",
                    "category": selected_category,
                    "tweet_id": tweet_id
                })


    # ---------- FLOOD INFRASTRUCTURE ----------
    elif tweet_type == "infrastructure":
        if today_infra_count >= INFRA_TWEETS_LIMIT:
            print(f"🚫 Reached daily infrastructure limit ({INFRA_TWEETS_LIMIT}).")
        else:
            tweet = generate_infrastructure_tweet()
            tweet_id = post_tweet(tweet)
            if tweet_id:
                today_infra_count += 1
                processed_articles.append({
                    "link": None,
                    "date": today,
                    "status": "posted",
                    "tweet": tweet,
                    "type": "infrastructure",
                    "tweet_id": tweet_id
                })

    elif tweet_type == "engagement":
        print("Engagement cycle — curating flood warnings & infrastructure tweets")
        process_mention_engagement()

    else:
        print("🤖 No tweet posted in this run (simulating human-like inactivity).")

    # save everything
    processed_articles = cleanup_old_articles(processed_articles)
    save_processed_articles(processed_articles)
    print("✅ floodlink_news.json updated.")
