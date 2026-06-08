# tagger.py
import json
import os
import hashlib
from groq import Groq
from config import GROQ_API_KEY

CACHE_FILE = "tags_cache.json"
_groq_client = Groq(api_key=GROQ_API_KEY)


def _file_fingerprint(folder_path: str, filename: str) -> str:
    """
    A stable key for each dataset. Uses filename + file size + last-modified time.
    If any of these change (file replaced/updated), it's treated as a new file.
    """
    fpath = os.path.join(folder_path, filename.split(" → ")[0])  # handles SQL → table_name entries
    try:
        stat = os.stat(fpath)
        raw = f"{filename}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()
    except FileNotFoundError:
        return hashlib.md5(filename.encode()).hexdigest()


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def generate_tags_for_profile(profile: dict) -> dict:
    """
    Calls Groq once per dataset and returns a tags dict:
    {
      "domain": "Financial Transactions / AML",
      "use_cases": ["fraud detection", "money laundering monitoring", ...],
      "keywords": ["transaction", "wire", "swift", "beneficiary", ...],
      "summary": "One sentence describing what this dataset is about."
    }
    """
    col_names = [c["name"] for c in profile.get("columns", [])][:40]
    prompt = f"""You are a data cataloging assistant. Given a dataset's column names, generate structured tags.

Dataset: "{profile['filename']}"
Columns: {", ".join(col_names)}
Rows: {profile['num_rows']}, Cols: {profile['num_cols']}
Inferred domain: {profile.get('inferred_domain', 'Unknown')}

Return ONLY a valid JSON object with these exact keys:
{{
  "domain": "short domain label",
  "use_cases": ["use case 1", "use case 2", "use case 3"],
  "keywords": ["kw1", "kw2", ... up to 25 keywords],
  "summary": "One sentence describing what this dataset is about."
}}

No extra text, no markdown, just the JSON."""

    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        import re
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        # Return a minimal fallback so the pipeline never breaks
        return {
            "domain": profile.get("inferred_domain", "Unknown"),
            "use_cases": [],
            "keywords": [c["name"].lower() for c in profile.get("columns", [])[:15]],
            "summary": f"Dataset with columns: {', '.join(col_names[:8])}",
        }


def get_tags_for_profiles(profiles: list, folder_path: str) -> tuple:
    """
    Main entry point called from app.py.
    
    For each profile:
      - If fingerprint is in cache → return cached tags instantly
      - If new/changed → call LLM, store in cache, save to disk
    
    Returns: (tags_map, new_tagged_count)
    """
    cache = load_cache()
    tags_map = {}
    cache_updated = False
    new_tagged_count = 0  # 1. Track the number of newly tagged files

    for profile in profiles:
        fname = profile["filename"]
        fingerprint = _file_fingerprint(folder_path, fname)
        cache_key = fingerprint  # key by content hash, not name

        if cache_key in cache:
            tags_map[fname] = cache[cache_key]["tags"]
        else:
            # New or modified file — generate tags
            tags = generate_tags_for_profile(profile)
            cache[cache_key] = {
                "filename": fname,
                "tags": tags,
            }
            tags_map[fname] = tags
            cache_updated = True
            new_tagged_count += 1  # 2. Increment counter on every LLM hit

    if cache_updated:
        save_cache(cache)

    return tags_map, new_tagged_count  # 3. Return both values as a tuple