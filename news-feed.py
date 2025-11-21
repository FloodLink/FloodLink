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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

XAI_MODEL = "grok-4-fast-reasoning"

# =========================================================
#                        TWITTER
# =========================================================

twitter_client = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET
)

# Accounts FloodLink may occasionally reply to (FILL IN REAL IDs)
TARGET_ACCOUNTS = {
    "sama": "1605",        # Replace with actual user IDs
    "elonmusk": "44196397",    # Replace with actual user IDs
    "stats_feed": "1335132884278108161",   # Replace with actual user IDs
    "balajis": "2178012643"
}

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
#                     STORAGE + LIMITS
# =========================================================

LOG_FILE = "floodlink_news.json"
REPLY_LOG_FILE = "floodlink_replies.json"

RETENTION_DAYS = 10
TWEET_THRESHOLD = 9  # 0–10 relevance; post only high-impact events

# Tweet type probabilities
RANDOM_NEWS = 0.3
RANDOM_STATISTIC = 0.1
RANDOM_INFRASTRUCTURE = 0.2
RANDOM_REPLY = 0.2
RANDOM_NONE = 0.2

# Daily tweet limits
NEWS_TWEETS_LIMIT = 4        # flood news
STAT_TWEETS_LIMIT = 1        # flood stats
INFRA_TWEETS_LIMIT = 1       # flood infrastructure
REPLY_TWEETS_LIMIT = 1       # replies

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
        and a.get("score", 0) >= TWEET_THRESHOLD
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

def select_tweet_type():
    return random.choices(
        ["news", "statistical", "infrastructure", "reply", "none"],
        [RANDOM_NEWS, RANDOM_STATISTIC, RANDOM_INFRASTRUCTURE, RANDOM_REPLY, RANDOM_NONE]
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
Summary: <one-sentence overview>

1. Item
2. Item
3. Item
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
    if os.path.exists(REPLY_LOG_FILE):
        with open(REPLY_LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_reply_log(log_data):
    with open(REPLY_LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def count_replies_today(reply_log):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return sum(1 for entry in reply_log.values() if entry["date"] == today)

def fetch_latest_tweets(user_id, max_results=5):
    try:
        tweets = twitter_client.get_users_tweets(
            id=user_id,
            max_results=max_results,
            tweet_fields=["id", "text", "created_at"],
            exclude=["retweets", "replies"]
        )
        return tweets.data if tweets.data else []
    except tweepy.errors.TweepyException as e:
        print(f"❌ Error fetching tweets for {user_id}: {e}")
        return []

def pick_most_recent_tweet(all_tweets, reply_log):
    new_tweets = [t for t in all_tweets if str(t.id) not in reply_log]
    if not new_tweets:
        print("🔍 No new tweets available to reply to.")
        return None
    return new_tweets[0]

def generate_grok_reply(tweet_text, username):
    """
    Reply as FloodLink with a short data / insight nugget about floods or extreme rainfall.
    """
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    prompt = f"""
You are replying as FloodLink, a flood-risk early warning system on X.

Read this tweet from @{username} (likely about climate, disasters, weather or resilience)
and reply with ONE concise, data-driven insight related to:

- floods, flash floods, storm surge, rainfall extremes, river levels,
- early warning systems, flood forecasting, or urban flood resilience.

Rules:
- Under 240 characters.
- NO hashtags.
- NO emojis except country flags before location names, if used.
- Tone: factual, calm, slightly analytical. No hype.

Tweet:
\"\"\"{tweet_text}\"\"\"

Your reply (text only, no username prefix):
"""

    response = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def reply_to_random_tweet():
    reply_log = load_reply_log()
    if count_replies_today(reply_log) >= REPLY_TWEETS_LIMIT:
        print(f"🚫 Reached daily reply limit ({REPLY_TWEETS_LIMIT}).")
        return

    if not TARGET_ACCOUNTS:
        print("⚠️ No TARGET_ACCOUNTS configured for FloodLink replies.")
        return

    username = random.choice(list(TARGET_ACCOUNTS.keys()))
    user_id = TARGET_ACCOUNTS[username]
    print(f"🔍 Fetching tweets from @{username}...")

    all_tweets = fetch_latest_tweets(user_id, max_results=5)
    if not all_tweets:
        return

    selected = pick_most_recent_tweet(all_tweets, reply_log)
    if not selected:
        return

    tweet_id = selected.id
    tweet_text = selected.text

    reply_text = generate_grok_reply(tweet_text, username)
    if not reply_text:
        print("❌ Failed to generate reply.")
        return

    try:
        twitter_client.create_tweet(
            text=f"@{username} {reply_text}",
            in_reply_to_tweet_id=tweet_id
        )
        print(f"✅ Replied to @{username}: {reply_text}")

        reply_log[str(tweet_id)] = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "username": username,
            "tweet_id": tweet_id
        }
        save_reply_log(reply_log)
    except tweepy.errors.TweepyException as e:
        print(f"❌ Error posting reply: {e}")

# =========================================================
#                      POSTING
# =========================================================

def post_tweet(tweet):
    print(f"🚀 Attempting to tweet: {tweet}")
    try:
        resp = twitter_client.create_tweet(text=tweet)
        print(f"✅ Tweet posted: {resp.data}")
        # small cooldown so runs don't spam
        time.sleep(60)
        return True
    except tweepy.errors.Forbidden as e:
        if "Status is a duplicate" in str(e):
            print("⚠️ Duplicate tweet detected. Skipping.")
        else:
            print(f"❌ Twitter API error: {e}")
        return False
    except tweepy.errors.TweepyException as e:
        print(f"❌ Other Tweepy error: {e}")
        return False

# =========================================================
#                        MAIN
# =========================================================

if __name__ == "__main__":
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
            if score >= TWEET_THRESHOLD:
                tweet = summarize_news(title, summary, source)
                if post_tweet(tweet):
                    today_news_count += 1
                    processed_articles.append({
                        "link": link,
                        "date": today,
                        "title": title,
                        "summary": summary,
                        "similarity_excluded": "No",
                        "score": score,
                        "status": "posted",
                        "tweet": tweet,
                        "type": "news"
                    })
            else:
                print(f"🚫 Article below threshold (score={score}): {title}")

    # ---------- FLOOD STATISTICS ----------
    elif tweet_type == "statistical":
        if today_stat_count >= STAT_TWEETS_LIMIT:
            print(f"🚫 Reached daily statistical limit ({STAT_TWEETS_LIMIT}).")
        else:
            selected_category = random.choice(STATISTICAL_CATEGORIES)
            tweet = generate_statistical_tweet(selected_category)
            if post_tweet(tweet):
                today_stat_count += 1
                processed_articles.append({
                    "link": None,
                    "date": today,
                    "status": "posted",
                    "tweet": tweet,
                    "type": "statistical",
                    "category": selected_category
                })

    # ---------- FLOOD INFRASTRUCTURE ----------
    elif tweet_type == "infrastructure":
        if today_infra_count >= INFRA_TWEETS_LIMIT:
            print(f"🚫 Reached daily infrastructure limit ({INFRA_TWEETS_LIMIT}).")
        else:
            tweet = generate_infrastructure_tweet()
            if post_tweet(tweet):
                today_infra_count += 1
                processed_articles.append({
                    "link": None,
                    "date": today,
                    "status": "posted",
                    "tweet": tweet,
                    "type": "infrastructure"
                })

    else:
        print("🤖 No tweet posted in this run (simulating human-like inactivity).")

    # save everything
    processed_articles = cleanup_old_articles(processed_articles)
    save_processed_articles(processed_articles)
    print("✅ floodlink_news.json updated.")
