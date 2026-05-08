import pandas as pd
import numpy as np
from collections import Counter


def profile_dataset(filename: str, df: pd.DataFrame) -> dict:
    """Extract a rich semantic + statistical profile from a DataFrame."""
    profile = {
        "filename": filename,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": [],
        "column_names_flat": [],
        "dtypes_summary": {},
        "has_datetime": False,
        "has_numeric": False,
        "has_categorical": False,
        "has_text": False,
        "has_id_cols": False,
        "has_amount_cols": False,
        "has_flag_cols": False,
        "inferred_domain": None,
    }

    dtype_counts = Counter()

    for col in df.columns:
        series = df[col]
        col_lower = col.lower().replace(" ", "_").replace("-", "_")

        dtype_str = str(series.dtype)
        null_pct = round(series.isnull().mean() * 100, 1)
        unique_count = series.nunique()
        sample_values = series.dropna().astype(str).unique()[:5].tolist()

        # Infer semantic type
        semantic_type = _infer_semantic_type(col_lower, series)

        col_info = {
            "name": col,
            "col_lower": col_lower,
            "dtype": dtype_str,
            "semantic_type": semantic_type,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "cardinality_ratio": round(unique_count / max(len(df), 1), 3),
            "sample_values": sample_values,
        }

        # Add numeric stats
        if pd.api.types.is_numeric_dtype(series):
            col_info["min"] = round(float(series.min()), 4) if not series.isnull().all() else None
            col_info["max"] = round(float(series.max()), 4) if not series.isnull().all() else None
            col_info["mean"] = round(float(series.mean()), 4) if not series.isnull().all() else None
            profile["has_numeric"] = True
            dtype_counts["numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile["has_datetime"] = True
            dtype_counts["datetime"] += 1
        elif unique_count <= 30 or (unique_count / max(len(df), 1) < 0.05):
            profile["has_categorical"] = True
            dtype_counts["categorical"] += 1
        else:
            profile["has_text"] = True
            dtype_counts["text"] += 1

        # Flag special columns
        if semantic_type == "id":
            profile["has_id_cols"] = True
        if semantic_type == "amount":
            profile["has_amount_cols"] = True
        if semantic_type in ("flag", "status", "risk_label"):
            profile["has_flag_cols"] = True

        profile["columns"].append(col_info)
        profile["column_names_flat"].append(col)

    profile["dtypes_summary"] = dict(dtype_counts)
    profile["inferred_domain"] = _infer_domain(profile)

    return profile


def _infer_semantic_type(col_lower: str, series: pd.Series) -> str:
    """Classify a column into a semantic type based on name + values."""
    id_keywords = ["id", "uuid", "key", "code", "ref", "account", "customer_id", "trans_id"]
    amount_keywords = ["amount", "balance", "salary", "income", "price", "revenue", "payment", "credit", "debit", "loan", "value"]
    date_keywords = ["date", "time", "timestamp", "dob", "birth", "created", "updated", "year", "month", "day"]
    flag_keywords = ["flag", "is_", "has_", "default", "churn", "fraud", "suspicious", "alert", "target", "label", "status"]
    risk_keywords = ["score", "rating", "risk", "fico", "cibil", "grade"]
    geo_keywords = ["country", "city", "state", "region", "zip", "postal", "address", "latitude", "longitude", "lat", "lon"]
    demo_keywords = ["age", "gender", "sex", "occupation", "education", "marital", "nationality", "ethnicity"]
    txn_keywords = ["transaction", "txn", "transfer", "wire", "swift", "iban", "bic", "sender", "receiver", "beneficiary"]
    segment_keywords = ["segment", "cluster", "tier", "category", "type", "class", "group"]

    for kw in id_keywords:
        if kw in col_lower:
            return "id"
    for kw in amount_keywords:
        if kw in col_lower:
            return "amount"
    for kw in date_keywords:
        if kw in col_lower:
            return "datetime"
    for kw in flag_keywords:
        if kw in col_lower:
            return "flag"
    for kw in risk_keywords:
        if kw in col_lower:
            return "risk_score"
    for kw in geo_keywords:
        if kw in col_lower:
            return "geography"
    for kw in demo_keywords:
        if kw in col_lower:
            return "demographic"
    for kw in txn_keywords:
        if kw in col_lower:
            return "transaction"
    for kw in segment_keywords:
        if kw in col_lower:
            return "segment"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _infer_domain(profile: dict) -> str:
    """Loosely infer the dataset's likely domain."""
    col_names = " ".join(profile["column_names_flat"]).lower()

    if any(k in col_names for k in ["loan", "fico", "cibil", "default", "credit_score", "repayment", "ltv", "delinquent"]):
        return "Credit / Lending"
    if any(k in col_names for k in ["transaction", "wire", "swift", "iban", "beneficiary", "suspicious", "aml", "anti_money"]):
        return "Financial Transactions / AML"
    if any(k in col_names for k in ["customer", "segment", "purchase", "recency", "frequency", "monetary", "clv", "churn"]):
        return "Customer / CRM"
    if any(k in col_names for k in ["patient", "diagnosis", "icd", "medication", "hospital"]):
        return "Healthcare"
    if any(k in col_names for k in ["product", "sku", "inventory", "order", "shipment"]):
        return "Retail / Supply Chain"
    if profile["has_amount_cols"] and profile["has_datetime"]:
        return "Financial"
    if profile["has_categorical"] and not profile["has_amount_cols"]:
        return "Behavioral / Demographic"
    return "General / Unknown"
