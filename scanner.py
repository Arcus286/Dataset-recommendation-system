"""
scanner.py
Scans two types of sources from a single folder:
  1. CSV / Excel files  — reads column headers
  2. SQL files          — parses CREATE TABLE statements, each table = one dataset entry
No database connection needed. SQL files are read as plain text.
"""

import os
import re
import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".sql")


def scan_folder(folder_path: str) -> list:
    datasets = []
    if not os.path.isdir(folder_path):
        return datasets

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            if fname.lower().endswith(".csv"):
                df = pd.read_csv(fpath, nrows=200)
                df = df.dropna(axis=1, how="all")
                datasets.append({"filename": fname, "df": df, "path": fpath, "error": None, "source": "file"})

            elif fname.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(fpath, nrows=200)
                df = df.dropna(axis=1, how="all")
                datasets.append({"filename": fname, "df": df, "path": fpath, "error": None, "source": "file"})

            elif fname.lower().endswith(".sql"):
                # Each table inside the SQL file becomes its own dataset entry
                tables = extract_tables_from_sql(fpath)
                if tables:
                    for table_name, columns in tables.items():
                        df = pd.DataFrame(columns=columns)
                        datasets.append({
                            "filename": f"{fname} → {table_name}",
                            "df": df,
                            "path": fpath,
                            "error": None,
                            "source": "sql",
                            "sql_table": table_name,
                        })
                else:
                    datasets.append({
                        "filename": fname, "df": None, "path": fpath,
                        "error": "No CREATE TABLE statements found", "source": "sql"
                    })

        except Exception as e:
            datasets.append({"filename": fname, "df": None, "path": fpath, "error": str(e), "source": "file"})

    return datasets


def extract_tables_from_sql(filepath: str) -> dict:
    """
    Reads a .sql file as plain text and extracts column names
    from every CREATE TABLE block found in the file.
    Returns { table_name: [col1, col2, ...] }
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    tables = {}

    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL
    )

    for match in pattern.finditer(content):
        table_name = match.group(1)
        body = match.group(2)
        columns = []

        for line in body.split("\n"):
            line = line.strip().rstrip(",").strip()
            if not line:
                continue
            upper = line.upper()
            if any(upper.startswith(kw) for kw in [
                "PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "INDEX",
                "KEY ", "CONSTRAINT", "CHECK", ")"
            ]):
                continue
            first_token = re.split(r"\s+", line)[0]
            col_name = first_token.strip("`\"[]'")
            if col_name and col_name.upper() not in {
                "PRIMARY", "FOREIGN", "UNIQUE", "INDEX",
                "KEY", "CONSTRAINT", "CHECK", "ENGINE", "DEFAULT"
            }:
                columns.append(col_name)

        if columns:
            tables[table_name] = columns

    return tables
