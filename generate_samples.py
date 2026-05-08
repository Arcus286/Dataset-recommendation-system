"""
generate_samples.py
Run this once to create sample datasets in a 'datasets' folder.
Generates 5 CSV files + 1 SQL file covering different domains.

Usage:
    python generate_samples.py
"""

import pandas as pd
import numpy as np
import os
import random

rng = np.random.default_rng(42)
random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(OUT, exist_ok=True)
N = 300


def save_csv(df, filename):
    path = os.path.join(OUT, filename)
    df.to_csv(path, index=False)
    print(f"  ✓  {filename:45s} ({len(df)} rows × {len(df.columns)} cols)")


# ── 1. Credit / Loan Applications ────────────────────────────────────────────
save_csv(pd.DataFrame({
    "application_id":     rng.integers(100000, 999999, N),
    "customer_age":       rng.integers(21, 65, N),
    "annual_income":      rng.integers(200000, 2000000, N),
    "loan_amount":        rng.integers(50000, 5000000, N),
    "loan_tenure_months": rng.integers(12, 240, N),
    "credit_score":       rng.integers(300, 900, N),
    "existing_debt":      rng.integers(0, 1000000, N),
    "employment_status":  rng.choice(["Salaried", "Self-Employed", "Unemployed"], N),
    "loan_type":          rng.choice(["Home", "Personal", "Vehicle", "Education"], N),
    "interest_rate":      rng.uniform(6.5, 18.0, N).round(2),
    "emi_amount":         rng.integers(2000, 50000, N),
    "collateral_value":   rng.integers(0, 10000000, N),
    "default_flag":       rng.choice([0, 1], N, p=[0.85, 0.15]),
}), "loan_applications.csv")


# ── 2. Customer Segmentation / RFM ───────────────────────────────────────────
segments = ["High Value", "At Risk", "New Customer", "Loyal", "Churned"]
save_csv(pd.DataFrame({
    "customer_id":         rng.integers(10000, 99999, N),
    "age":                 rng.integers(18, 70, N),
    "gender":              rng.choice(["Male", "Female", "Other"], N),
    "city":                rng.choice(["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"], N),
    "recency_days":        rng.integers(1, 365, N),
    "purchase_frequency":  rng.integers(1, 52, N),
    "monetary_value":      rng.integers(500, 100000, N),
    "avg_order_value":     rng.integers(200, 5000, N),
    "preferred_channel":   rng.choice(["Online", "In-Store", "Mobile App"], N),
    "loyalty_tier":        rng.choice(["Bronze", "Silver", "Gold", "Platinum"], N),
    "churn_probability":   rng.uniform(0, 1, N).round(3),
    "tenure_months":       rng.integers(1, 120, N),
    "segment":             rng.choice(segments, N),
}), "customer_segments.csv")


# ── 3. Financial Transactions / AML ──────────────────────────────────────────
channels = ["NEFT", "RTGS", "IMPS", "UPI", "SWIFT", "WIRE"]
save_csv(pd.DataFrame({
    "transaction_id":    [f"TXN{rng.integers(1000000,9999999)}" for _ in range(N)],
    "sender_account":    rng.integers(10000000, 99999999, N),
    "receiver_account":  rng.integers(10000000, 99999999, N),
    "transaction_amount": rng.uniform(100, 1000000, N).round(2),
    "currency":          rng.choice(["INR", "USD", "EUR", "GBP", "AED"], N),
    "transaction_date":  pd.date_range("2023-01-01", periods=N, freq="2h"),
    "sender_country":    rng.choice(["IN", "US", "AE", "NG", "CH", "SG"], N),
    "receiver_country":  rng.choice(["IN", "US", "AE", "NG", "CH", "SG"], N),
    "channel":           rng.choice(channels, N),
    "merchant_category": rng.choice(["Retail", "Crypto", "Forex", "Gaming", "Transfer"], N),
    "risk_score":        rng.uniform(0, 1, N).round(3),
    "is_flagged":        rng.choice([0, 1], N, p=[0.93, 0.07]),
    "alert_type":        rng.choice(["None", "Structuring", "Smurfing", "Rapid Movement", "None"], N),
}), "financial_transactions.csv")


# ── 4. HR / Employee Data ────────────────────────────────────────────────────
save_csv(pd.DataFrame({
    "employee_id":       rng.integers(1000, 9999, N),
    "first_name":        [f"Emp_{i}" for i in range(N)],
    "department":        rng.choice(["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"], N),
    "job_title":         rng.choice(["Analyst", "Manager", "Senior Engineer", "Executive", "Associate"], N),
    "years_experience":  rng.integers(0, 20, N),
    "monthly_salary":    rng.integers(20000, 200000, N),
    "performance_rating": rng.choice(["Excellent", "Good", "Average", "Poor"], N),
    "work_location":     rng.choice(["Remote", "On-site", "Hybrid"], N),
    "hire_date":         pd.date_range("2010-01-01", periods=N, freq="7D"),
    "attrition":         rng.choice(["Yes", "No"], N, p=[0.18, 0.82]),
    "training_hours":    rng.integers(0, 100, N),
    "satisfaction_score": rng.uniform(1, 5, N).round(1),
}), "hr_employee_data.csv")


# ── 5. E-commerce / Sales ────────────────────────────────────────────────────
save_csv(pd.DataFrame({
    "order_id":          [f"ORD{rng.integers(100000,999999)}" for _ in range(N)],
    "product_id":        [f"PRD{rng.integers(1000,9999)}" for _ in range(N)],
    "product_category":  rng.choice(["Electronics", "Fashion", "Grocery", "Books", "Home", "Sports"], N),
    "product_name":      [f"Product_{rng.integers(1,200)}" for _ in range(N)],
    "quantity_sold":     rng.integers(1, 50, N),
    "unit_price":        rng.uniform(50, 5000, N).round(2),
    "discount_pct":      rng.uniform(0, 40, N).round(1),
    "total_revenue":     rng.uniform(100, 100000, N).round(2),
    "order_date":        pd.date_range("2023-01-01", periods=N, freq="3h"),
    "delivery_status":   rng.choice(["Delivered", "Pending", "Cancelled", "Returned"], N),
    "customer_rating":   rng.choice([1, 2, 3, 4, 5], N),
    "return_flag":       rng.choice([0, 1], N, p=[0.88, 0.12]),
    "sales_region":      rng.choice(["North", "South", "East", "West", "Central"], N),
}), "ecommerce_sales.csv")


# ── 6. SQL file — Banking System ─────────────────────────────────────────────
sql_content = """-- Banking System Database
-- Drop and recreate for fresh setup
DROP DATABASE IF EXISTS banking_system;
CREATE DATABASE banking_system;
USE banking_system;

-- Customer personal and demographic info
CREATE TABLE customers (
    customer_id       INT PRIMARY KEY,
    first_name        VARCHAR(50),
    last_name         VARCHAR(50),
    date_of_birth     DATE,
    gender            VARCHAR(10),
    email             VARCHAR(100),
    phone             VARCHAR(15),
    address           VARCHAR(255),
    city              VARCHAR(50),
    state             VARCHAR(50),
    country           VARCHAR(50),
    kyc_status        VARCHAR(20),
    created_at        DATETIME
);

-- Bank accounts linked to customers
CREATE TABLE accounts (
    account_id        INT PRIMARY KEY,
    customer_id       INT,
    account_number    VARCHAR(20),
    account_type      VARCHAR(50),
    balance           DECIMAL(15,2),
    currency          VARCHAR(10),
    opened_date       DATE,
    status            VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- All financial transactions
CREATE TABLE transactions (
    transaction_id        INT PRIMARY KEY,
    account_id            INT,
    transaction_type      VARCHAR(20),
    amount                DECIMAL(15,2),
    transaction_timestamp DATETIME,
    merchant_name         VARCHAR(100),
    merchant_category     VARCHAR(50),
    sender_country        VARCHAR(10),
    receiver_country      VARCHAR(10),
    channel               VARCHAR(30),
    status                VARCHAR(20),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Credit card details
CREATE TABLE credit_cards (
    card_id           INT PRIMARY KEY,
    customer_id       INT,
    account_id        INT,
    card_type         VARCHAR(50),
    credit_limit      DECIMAL(15,2),
    outstanding_balance DECIMAL(15,2),
    issued_date       DATE,
    expiry_date       DATE,
    status            VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Loan records
CREATE TABLE loans (
    loan_id           INT PRIMARY KEY,
    customer_id       INT,
    loan_type         VARCHAR(50),
    loan_amount       DECIMAL(15,2),
    interest_rate     DECIMAL(5,2),
    tenure_months     INT,
    emi_amount        DECIMAL(15,2),
    outstanding_principal DECIMAL(15,2),
    loan_status       VARCHAR(20),
    start_date        DATE,
    end_date          DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Fraud and AML alerts
CREATE TABLE fraud_alerts (
    alert_id          INT PRIMARY KEY,
    transaction_id    INT,
    risk_score        DECIMAL(5,2),
    alert_type        VARCHAR(50),
    alert_status      VARCHAR(20),
    flagged_reason    VARCHAR(255),
    reviewed_by       VARCHAR(100),
    created_at        DATETIME,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);
"""

sql_path = os.path.join(OUT, "banking_system.sql")
with open(sql_path, "w") as f:
    f.write(sql_content)
print(f"  ✓  {'banking_system.sql':45s} (6 tables: customers, accounts, transactions, credit_cards, loans, fraud_alerts)")

print()
print(f"All sample files saved to: {OUT}")
print()
print("Now open the app, enter this folder path:")
print(f"  {OUT}")
print()
print("Try these sample queries:")
print("  - 'I want to detect suspicious money transfers and fraud'")
print("  - 'customer purchase behavior and loyalty segmentation'")
print("  - 'loan repayment risk and credit score analysis'")
print("  - 'employee performance and salary trends'")
print("  - 'product sales revenue and return rates'")
