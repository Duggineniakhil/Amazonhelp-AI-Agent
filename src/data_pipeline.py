"""
Data pipeline for AmazonHelp customer support conversations.

Responsibilities:
1. Load raw CSV from Kaggle dataset
2. Identify AmazonHelp's author_id
3. Filter to AmazonHelp conversations only
4. Reconstruct multi-turn conversation threads
5. Clean tweet text (remove mentions, URLs, placeholders)
6. Subsample and export as structured JSON

Output: data/processed/amazon_threads.json
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_CSV_PATH,
    THREADS_JSON_PATH,
    PROCESSED_DATA_DIR,
    BRAND_NAME,
    MAX_THREADS,
    MIN_THREAD_LENGTH,
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw TWCS CSV file."""
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {RAW_CSV_PATH}. "
            "Run 'python scripts/download_data.py' first."
        )
    
    print(f"[→] Loading raw CSV from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"    Loaded {len(df):,} tweets")
    print(f"    Columns: {list(df.columns)}")
    return df


def identify_brand_author(df: pd.DataFrame) -> str:
    """
    Identify AmazonHelp's author_id.
    
    Strategy: Look at outbound tweets (inbound=False) whose text starts with
    or contains @AmazonHelp-like patterns. The most common author_id among
    these is the brand's account.
    """
    print(f"\n[→] Identifying {BRAND_NAME} author_id...")
    
    # Outbound tweets (from brands, not customers)
    outbound = df[df["inbound"] == False].copy()
    print(f"    Total outbound tweets: {len(outbound):,}")
    
    # In this dataset, brand names appear in tweet text as @mentions
    # Look for tweets mentioning "amazon" in the author's text patterns
    # The brand author is the one who responds the most
    
    # Method 1: Check if text contains brand-like mentions
    # In the TWCS dataset, the author_id for brands is typically the brand handle itself
    brand_patterns = ["amazonhelp", "amazon_help", "amazon"]
    
    # The dataset may have author_id as the actual Twitter handle
    # Let's check the top outbound authors
    top_authors = outbound["author_id"].value_counts().head(20)
    print(f"    Top 5 outbound authors: {dict(top_authors.head())}")
    
    # Look for AmazonHelp in author_ids (case-insensitive)
    for author_id in top_authors.index:
        author_str = str(author_id).lower()
        if any(pat in author_str for pat in brand_patterns):
            print(f"    [✓] Found brand author_id: {author_id}")
            return str(author_id)
    
    # Method 2: If author_ids are numeric/anonymized, look in tweet text
    # Brand replies often start with "@username" — look for AmazonHelp in outbound text
    for author_id in top_authors.index:
        sample_texts = outbound[outbound["author_id"] == author_id]["text"].head(5).tolist()
        # Check if these look like Amazon support responses
        amazon_keywords = ["order", "ship", "deliver", "refund", "account", "DM", "help"]
        keyword_count = sum(
            1 for text in sample_texts 
            for kw in amazon_keywords 
            if kw.lower() in str(text).lower()
        )
        if keyword_count >= 5:  # At least 5 keyword hits in 5 samples
            print(f"    [✓] Identified likely brand author_id by content: {author_id}")
            return str(author_id)
    
    # Fallback: just use the most prolific outbound author
    fallback = str(top_authors.index[0])
    print(f"    [⚠] Using fallback (most prolific outbound author): {fallback}")
    return fallback


def filter_brand_conversations(df: pd.DataFrame, brand_author_id: str) -> pd.DataFrame:
    """Filter dataset to only include conversations involving the brand."""
    print(f"\n[→] Filtering conversations for brand author: {brand_author_id}")
    
    # Get all tweet_ids where the brand is the author (brand replies)
    brand_tweets = df[df["author_id"].astype(str) == brand_author_id]
    print(f"    Brand tweets: {len(brand_tweets):,}")
    
    # Get customer tweets that the brand responded to
    # Need to convert to int first to handle cases like 272.0 != 272
    brand_response_to = brand_tweets["in_response_to_tweet_id"].dropna().astype(float).astype(int).astype(str).unique()
    
    # Get tweets that responded to brand tweets
    brand_tweet_ids = set(brand_tweets["tweet_id"].astype(str).unique())
    
    # Collect all tweet IDs in brand conversations
    relevant_tweet_ids = brand_tweet_ids.copy()
    relevant_tweet_ids.update(brand_response_to)
    
    # Also get customer tweets that the brand's responses link to
    # via response_tweet_id column
    for _, row in brand_tweets.iterrows():
        resp_ids = str(row.get("response_tweet_id", ""))
        if resp_ids and resp_ids != "nan":
            for rid in resp_ids.split(","):
                relevant_tweet_ids.add(rid.strip())
    
    # Filter to relevant tweets
    filtered = df[df["tweet_id"].astype(str).isin(relevant_tweet_ids)].copy()
    print(f"    Relevant tweets (brand + their customers): {len(filtered):,}")
    
    return filtered


def clean_text(text: str) -> str:
    """
    Clean a tweet's text for processing.
    
    - Remove @mentions (but keep the rest)
    - Replace __url__ placeholders
    - Replace __email__ placeholders
    - Remove excessive whitespace
    - Keep emoji and punctuation (they carry sentiment)
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove @mentions at the start of tweets (common in replies)
    text = re.sub(r"^(@\w+\s*)+", "", text).strip()
    
    # Replace platform placeholders
    text = text.replace("__url__", "[link]")
    text = text.replace("__email__", "[email]")
    
    # Remove remaining @mentions within text (but keep the info)
    text = re.sub(r"@(\w+)", r"\1", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def reconstruct_threads(df: pd.DataFrame, brand_author_id: str) -> list:
    """
    Reconstruct multi-turn conversation threads.
    
    A thread is a sequence of messages between a customer and AmazonHelp.
    We use in_response_to_tweet_id to build the reply chain.
    
    Returns a list of thread dicts:
    {
        "thread_id": str,
        "messages": [
            {"role": "customer"|"agent", "text": str, "tweet_id": str}
        ]
    }
    """
    print("\n[→] Reconstructing conversation threads...")
    
    # Build lookup maps
    tweet_map = {}
    response_map = defaultdict(list)  # tweet_id → list of response tweet_ids
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Building tweet map"):
        tid = str(row["tweet_id"])
        resp_to_raw = row.get("in_response_to_tweet_id")
        if pd.isna(resp_to_raw):
            in_response_to = "nan"
        else:
            try:
                in_response_to = str(int(float(resp_to_raw)))
            except ValueError:
                in_response_to = str(resp_to_raw)
        
        tweet_map[tid] = {
            "tweet_id": tid,
            "author_id": str(row["author_id"]),
            "text": str(row.get("text", "")),
            "inbound": row.get("inbound", True),
            "in_response_to": in_response_to,
        }
        
        # Build response chain
        if in_response_to != "nan":
            response_map[in_response_to].append(tid)
    
    # Find thread roots (tweets with no in_response_to that are inbound/customer messages)
    roots = []
    for tid, tweet in tweet_map.items():
        if tweet["in_response_to"] == "nan" and tweet["inbound"] == True:
            # Check if this tweet got a response from the brand
            responses = response_map.get(tid, [])
            brand_responded = any(
                tweet_map.get(r, {}).get("author_id") == brand_author_id
                for r in responses
            )
            if brand_responded:
                roots.append(tid)
    
    print(f"    Found {len(roots):,} conversation roots")
    
    # Build threads by following the reply chain
    threads = []
    for root_id in tqdm(roots, desc="   Building threads"):
        thread_messages = []
        
        # BFS through the reply chain
        queue = [root_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited or current_id not in tweet_map:
                continue
            visited.add(current_id)
            
            tweet = tweet_map[current_id]
            role = "agent" if tweet["author_id"] == brand_author_id else "customer"
            cleaned = clean_text(tweet["text"])
            
            if cleaned:  # Skip empty messages
                thread_messages.append({
                    "role": role,
                    "text": cleaned,
                    "tweet_id": current_id,
                })
            
            # Follow responses
            for resp_id in response_map.get(current_id, []):
                if resp_id not in visited:
                    queue.append(resp_id)
        
        # Only keep threads with meaningful exchanges
        if len(thread_messages) >= MIN_THREAD_LENGTH:
            # Ensure thread starts with customer and has both roles
            roles = set(m["role"] for m in thread_messages)
            if "customer" in roles and "agent" in roles:
                threads.append({
                    "thread_id": root_id,
                    "messages": thread_messages,
                })
    
    print(f"    Reconstructed {len(threads):,} valid threads")
    return threads


def subsample_threads(threads: list) -> list:
    """Subsample threads if we have more than MAX_THREADS."""
    if len(threads) <= MAX_THREADS:
        return threads
    
    import random
    random.seed(42)  # Reproducible sampling
    sampled = random.sample(threads, MAX_THREADS)
    print(f"\n[→] Subsampled {MAX_THREADS:,} threads from {len(threads):,}")
    return sampled


def compute_stats(threads: list):
    """Print summary statistics about the processed threads."""
    total_messages = sum(len(t["messages"]) for t in threads)
    customer_msgs = sum(1 for t in threads for m in t["messages"] if m["role"] == "customer")
    agent_msgs = sum(1 for t in threads for m in t["messages"] if m["role"] == "agent")
    avg_length = total_messages / len(threads) if threads else 0
    
    print(f"\n[✓] Thread Statistics:")
    print(f"    Total threads:       {len(threads):,}")
    print(f"    Total messages:      {total_messages:,}")
    print(f"    Customer messages:   {customer_msgs:,}")
    print(f"    Agent messages:      {agent_msgs:,}")
    print(f"    Avg thread length:   {avg_length:.1f} messages")
    
    # Message length distribution
    customer_lengths = [
        len(m["text"]) for t in threads for m in t["messages"] if m["role"] == "customer"
    ]
    agent_lengths = [
        len(m["text"]) for t in threads for m in t["messages"] if m["role"] == "agent"
    ]
    
    if customer_lengths:
        print(f"    Avg customer msg:    {sum(customer_lengths)/len(customer_lengths):.0f} chars")
    if agent_lengths:
        print(f"    Avg agent msg:       {sum(agent_lengths)/len(agent_lengths):.0f} chars")


def save_threads(threads: list):
    """Save processed threads to JSON."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(THREADS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    
    size_mb = THREADS_JSON_PATH.stat().st_size / 1e6
    print(f"\n[✓] Saved {len(threads):,} threads to {THREADS_JSON_PATH} ({size_mb:.1f} MB)")


def load_threads() -> list:
    """Load processed threads from JSON."""
    if not THREADS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Processed threads not found at {THREADS_JSON_PATH}. "
            "Run the data pipeline first."
        )
    
    with open(THREADS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pairs(threads: list) -> list:
    """
    Extract (customer_message, agent_reply) pairs from threads.
    
    Each pair represents one customer message and the immediate agent response.
    Used for retrieval index building and evaluation.
    """
    pairs = []
    for thread in threads:
        messages = thread["messages"]
        for i, msg in enumerate(messages):
            if msg["role"] == "customer":
                # Find the next agent message
                for j in range(i + 1, len(messages)):
                    if messages[j]["role"] == "agent":
                        pairs.append({
                            "thread_id": thread["thread_id"],
                            "customer_text": msg["text"],
                            "agent_text": messages[j]["text"],
                            "customer_tweet_id": msg["tweet_id"],
                            "agent_tweet_id": messages[j]["tweet_id"],
                        })
                        break
    return pairs


def run_pipeline():
    """Run the full data pipeline."""
    print("=" * 60)
    print("  AmazonHelp Data Pipeline")
    print("=" * 60)
    
    # Step 1: Load raw data
    df = load_raw_data()
    
    # Step 2: Identify brand
    brand_author_id = identify_brand_author(df)
    
    # Step 3: Filter conversations
    filtered = filter_brand_conversations(df, brand_author_id)
    
    # Step 4: Reconstruct threads
    threads = reconstruct_threads(filtered, brand_author_id)
    
    # Step 5: Subsample
    threads = subsample_threads(threads)
    
    # Step 6: Stats
    compute_stats(threads)
    
    # Step 7: Extract and save pairs summary
    pairs = extract_pairs(threads)
    print(f"    Extracted pairs:     {len(pairs):,}")
    
    # Save pairs separately for quick access
    pairs_path = PROCESSED_DATA_DIR / "amazon_pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    # Step 8: Save
    save_threads(threads)
    
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    
    return threads


if __name__ == "__main__":
    run_pipeline()
