# Insight-to-Innovation Engine

A lightweight Python Flask web application that converts consumer review CSVs into **exactly 5** ranked product innovation opportunities, then generates a **Strategic Activation Plan** for each opportunity.

## What it does

Workflow: **Upload → Analyze → Rank → Activate Strategy**

From your CSV it will:

- Filter negative reviews (**rating ≤ 3**)
- Clean and preprocess review text
- Detect recurring complaint patterns using NLP clustering
- Rank opportunities with a simple 20-point scoring system
- Generate **5 opportunity cards** (no keyword clusters shown)
- Generate an on-demand **Strategic Activation Plan** per opportunity

## CSV format

Required columns:

- `rating`
- `review_text`

Optional columns:

- `product_name`
- `brand_name`
- `category`

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python app.py
```

Open in your browser:

`http://127.0.0.1:5000`

