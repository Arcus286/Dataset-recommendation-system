# 🔎 DataLens — AI-Powered Dataset Discovery

DataLens helps you find the most relevant datasets for your use case using a three-stage AI pipeline. Just describe what you're looking for in plain English — no exact keywords needed.

> *"unusual money movement"* → correctly finds fraud and AML transaction datasets  
> *"predict who might leave"* → correctly finds employee attrition and churn datasets

---

## How It Works

**Stage 1 — LLM Query Expansion (Groq)**  
Your query is sent to an LLM which expands it into domain-aware terms and likely column name patterns — even concepts you didn't explicitly mention.

**Stage 2 — TF-IDF + Keyword Overlap**  
Fast structural matching runs on the LLM-expanded query across all dataset column names.

**Stage 3 — LLM Re-ranking (Groq)**  
Top candidates are scored semantically by the LLM with a one-line reason per dataset. Final score blends both stages.

---

## Supported File Formats

| Format | Details |
|--------|---------|
| 📄 CSV | Reads column headers + 200 rows |
| 📊 Excel | `.xlsx` and `.xls` |
| 🗄️ SQL | Parses `CREATE TABLE` statements — each table is treated as a separate dataset |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/datalens.git
cd datalens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Groq API key
- Sign up free at [console.groq.com](https://console.groq.com)
- Go to **API Keys** → **Create API Key**
- Copy the key

### 4. Create a `.env` file in the project root
```
GROQ_API_KEY=gsk_your_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
datalens/
├── app.py            # Streamlit UI
├── matcher.py        # Three-stage AI matching pipeline
├── profiler.py       # Extracts column stats and domain info from datasets
├── scanner.py        # Scans folder for CSV, Excel, and SQL files
├── config.py         # Loads API key from .env
├── requirements.txt
└── .env              # Your Groq API key (never committed)
```

---

## Usage

1. Point DataLens at a folder containing your datasets
2. Type a natural language query describing what you need
3. Hit **Find Matching Datasets**
4. Results are ranked by relevance score (0–100) with matched columns and LLM reasoning shown per dataset

---

## Scoring

Each dataset gets a score out of 100:

| Score | Meaning |
|-------|---------|
| 🟢 40+ | Strong match |
| 🟡 20–39 | Weak match |
| ⚫ < 20 | Likely irrelevant |

Final score = **35% TF-IDF** + **65% LLM semantic score**

---

## Requirements

- Python 3.8+
- Groq API key (free tier is sufficient — 14,400 requests/day)

---

## Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## License

MIT
