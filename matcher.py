"""
matcher.py
Two-stage matching pipeline — LLM is mandatory, not optional.

  Stage 1 — LLM Query Expansion (Groq)
      The user's raw query is sent to the LLM first. It returns a rich,
      intent-aware expansion: domain terms, relevant column name patterns,
      and synonyms — even when the user's words don't literally match anything.

  Stage 2 — TF-IDF + Keyword Overlap (fast structural filter)
      Runs on the LLM-expanded query, not the raw query. This means even
      vague queries like "unusual money movement" produce accurate column hits.

  Stage 3 — LLM Re-ranking (Groq)
      Top candidates are sent back to the LLM for semantic scoring with a
      short reason per dataset. Blended with TF-IDF for the final score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import GROQ_API_KEY
from groq import Groq

_groq_client = Groq(api_key=GROQ_API_KEY)

# ── Fallback synonyms (used if LLM expansion fails) ──────────────────────────
SYNONYMS = {
    "customer":    ["customer", "client", "user", "buyer", "consumer", "account"],
    "purchase":    ["purchase", "buy", "order", "transaction", "spend", "bought"],
    "behavior":    ["behavior", "activity", "pattern", "frequency", "recency", "engagement"],
    "spending":    ["spending", "spend", "monetary", "revenue", "amount", "value", "price"],
    "segment":     ["segment", "cluster", "group", "tier", "category", "cohort"],
    "loyalty":     ["loyalty", "retention", "churn", "clv", "lifetime"],
    "loan":        ["loan", "credit", "mortgage", "debt", "borrower", "lending", "balance"],
    "default":     ["default", "delinquent", "risk", "flag", "repayment"],
    "risk":        ["risk", "score", "rating", "probability", "grade"],
    "income":      ["income", "salary", "annual", "earning", "wage"],
    "fraud":       ["fraud", "suspicious", "alert", "aml", "laundering", "anomaly"],
    "transaction": ["transaction", "transfer", "payment", "wire", "txn", "sender", "receiver"],
    "money":       ["money", "amount", "currency", "financial", "fund", "cash"],
    "medical":     ["medical", "health", "hospital", "patient", "diagnosis", "clinical"],
    "employee":    ["employee", "staff", "hr", "salary", "department", "attrition"],
    "sales":       ["sales", "revenue", "order", "product", "quantity"],
    "location":    ["location", "city", "region", "country", "address", "geography"],
    "time":        ["date", "timestamp", "time", "year", "month", "day"],
}


def _fallback_expand(query: str) -> str:
    """Static synonym expansion — used only if LLM expansion fails."""
    words = query.lower().split()
    expanded = set(words)
    for word in words:
        for key, synonyms in SYNONYMS.items():
            if word in synonyms or word == key:
                expanded.update(synonyms)
    return " ".join(expanded)


def llm_expand_query(query: str) -> str:
    """
    Stage 1: Ask the LLM to rewrite the user's query into a rich set of
    domain-aware terms and likely column name patterns.

    This is what makes the system understand "unusual money movement"
    and know to look for: transaction, amount, transfer, wire, suspicious,
    aml, sender, receiver, beneficiary, flag, anomaly, etc.
    """
    prompt = f"""You are a data discovery assistant. A user is searching for datasets using a natural language query.

Your job: expand the query into a comprehensive list of technical terms, column name patterns, and domain-specific keywords that would appear in relevant datasets — even if those words aren't in the original query.

User query: "{query}"

Return ONLY a flat list of space-separated lowercase keywords. No explanations, no bullets, no JSON. Just keywords.

Think about:
- What domain does this query belong to? (finance, HR, healthcare, retail, etc.)
- What column names would a relevant dataset likely have?
- What synonyms or technical terms map to the user's intent?
- What related concepts are implied but not stated?

Output example format (just keywords, nothing else):
transaction amount transfer wire suspicious aml sender receiver beneficiary flag anomaly fraud balance account"""

    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        expanded = response.choices[0].message.content.strip().lower()
        # Merge original query words too so nothing is lost
        original_words = query.lower().split()
        all_terms = set(expanded.split() + original_words)
        return " ".join(all_terms)
    except Exception as e:
        # LLM failed — fall back to static synonym expansion
        print(f"[LLM expand failed, using fallback]: {e}")
        return _fallback_expand(query)


def score_all(profiles: list, user_query: str) -> list:
    """
    Full three-stage pipeline:
      1. LLM query expansion
      2. TF-IDF + keyword overlap on the expanded query
      3. LLM re-ranking of top candidates
    """
    # Ensure col_text exists for every profile
    for p in profiles:
        if "col_text" not in p or not p["col_text"].strip():
            p["col_text"] = " ".join(
                c["name"].lower().replace("_", " ").replace("-", " ")
                for c in p.get("columns", [])
            )

    # ── Stage 1: LLM query expansion ─────────────────────────────────────────
    expanded_query = llm_expand_query(user_query)
    query_words = set(expanded_query.split())

    # ── Stage 2: TF-IDF + keyword overlap ────────────────────────────────────
    corpus = [expanded_query] + [p["col_text"] for p in profiles]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cos_sims = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

    results = []
    for i, profile in enumerate(profiles):
        tfidf_score = float(cos_sims[i])
        col_words = set(profile["col_text"].lower().split())
        overlap = len(query_words & col_words)
        overlap_score = min(overlap / max(len(set(user_query.lower().split())), 1), 1.0)

        raw = (0.55 * tfidf_score + 0.45 * overlap_score) * 100
        final_score = min(100, round(raw, 1))

        results.append({
            **profile,
            "score": final_score,
            "tfidf_score": round(tfidf_score * 100, 1),
            "overlap_score": round(overlap_score * 100, 1),
            "llm_score": None,
            "llm_reason": "",
            "recommended": final_score >= 20,
            "matched_columns": get_matched_columns(profile, query_words),
            "expanded_query": expanded_query,
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # ── Stage 3: LLM re-ranking ───────────────────────────────────────────────
    results = groq_rerank(results, user_query)

    return results


def get_matched_columns(profile: dict, query_words: set) -> list:
    matched = []
    for col in profile.get("columns", []):
        col_words = set(col["name"].lower().replace("_", " ").replace("-", " ").split())
        if col_words & query_words:
            matched.append(col["name"])
    return matched[:10]


def groq_rerank(results: list, user_query: str) -> list:
    """
    Stage 3: Send the top 15 dataset summaries to the LLM and get a
    semantic relevance score (0–100) with a one-line reason for each.
    Final score = 35% TF-IDF + 65% LLM.
    """
    try:
        dataset_summaries = []
        for i, r in enumerate(results[:15]):
            cols = [c["name"] for c in r.get("columns", [])]
            dataset_summaries.append(
                f'{i+1}. "{r["filename"]}" — columns: {", ".join(cols[:20])}'
            )

        prompt = f"""You are a data analyst helping find relevant datasets.

User query: "{user_query}"

Below are dataset descriptions (name + column names). Score each from 0 to 100 on how relevant it is to the query. Consider what the data is actually about — not just literal keyword matches. A dataset about "AML transaction monitoring" is very relevant to "unusual money movement" even if those exact words aren't in it.

Datasets:
{chr(10).join(dataset_summaries)}

Respond ONLY with a JSON array, one object per dataset, in the same order:
[
  {{"rank": 1, "score": 85, "reason": "one short sentence why"}},
  {{"rank": 2, "score": 40, "reason": "one short sentence why"}},
  ...
]
No extra text, just the JSON array."""

        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )

        import json, re
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        llm_scores = json.loads(raw)

        for i, item in enumerate(llm_scores):
            if i >= len(results):
                break
            llm_s = float(item.get("score", 0))
            tfidf_s = results[i]["score"]
            blended = round(0.35 * tfidf_s + 0.65 * llm_s, 1)
            results[i]["llm_score"] = llm_s
            results[i]["llm_reason"] = item.get("reason", "")
            results[i]["score"] = blended
            results[i]["recommended"] = blended >= 25

        results.sort(key=lambda x: x["score"], reverse=True)

    except Exception as e:
        # Re-ranking failed — TF-IDF scores stand, mark reason
        for r in results:
            r["llm_reason"] = f"LLM re-rank unavailable: {str(e)[:80]}"

    return results


def find_similar_datasets(profiles: list, threshold: float = 0.75) -> list:
    if len(profiles) < 2:
        return []
    for p in profiles:
        if "col_text" not in p or not p["col_text"].strip():
            p["col_text"] = " ".join(
                c["name"].lower().replace("_", " ")
                for c in p.get("columns", [])
            )
    corpus = [p["col_text"] for p in profiles]
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(matrix)
    pairs = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            if sim[i][j] >= threshold:
                pairs.append({
                    "dataset_a": profiles[i]["filename"],
                    "dataset_b": profiles[j]["filename"],
                    "similarity": round(float(sim[i][j]) * 100, 1),
                })
    return sorted(pairs, key=lambda x: x["similarity"], reverse=True)
