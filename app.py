import re
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = "insight-to-innovation-engine-dev"

REQUIRED_COLUMNS = {"rating", "review_text"}

UPLOAD_STORE: Dict[str, List["Opportunity"]] = {}


@dataclass
class Opportunity:
    opp_id: str
    theme: str
    category_hint: str
    product_name: str
    product_concept: str
    target_persona: str
    core_problem: str
    formulation_direction: str
    product_format: str
    suggested_price_band: str
    key_differentiation: str
    why_competitors_fail: str
    opportunity_score: int
    evidence_count: int
    sample_quotes: List[str]


def _safe_str(x):
    if x is None:
        return ""
    return str(x)


def _clean_text(text: str):
    text = _safe_str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sample_quotes(texts: List[str], k=2):
    out = []
    for t in texts[:10]:
        s = _safe_str(t).strip()
        if len(s) > 140:
            s = s[:140] + "..."
        out.append(s)
        if len(out) >= k:
            break
    return out


def _cluster_negative_reviews(df: pd.DataFrame):
    neg = df[df["rating"] <= 3].copy()

    if len(neg) > 200:
        neg = neg.sample(200, random_state=42)

    neg["clean_text"] = neg["review_text"].map(_clean_text)
    neg = neg[neg["clean_text"].str.len() >= 8]

    if neg.empty:
        return neg, np.array([]), None

    docs = neg["clean_text"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=6000,
        min_df=(2 if len(docs) >= 30 else 1),
    )

    X = vectorizer.fit_transform(docs)

    k = int(np.clip(round(np.sqrt(max(2, len(docs) / 2))), 2, 8))
    if len(docs) < 10:
        k = 2
    if len(docs) < 4:
        k = 1

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    neg["cluster"] = labels
    return neg, labels, model


def _generate_opportunities(df: pd.DataFrame):

    neg, labels, _ = _cluster_negative_reviews(df)

    if neg.empty:
        return []

    groups = neg.groupby("cluster")
    opps = []

    for idx, (cl, g) in enumerate(groups, start=1):

        texts = g["review_text"].tolist()

        opps.append(
            Opportunity(
                opp_id=str(uuid.uuid4()),
                theme="Customer Pain Point",
                category_hint="General",
                product_name=f"Innovation Concept {idx}",
                product_concept="Concept derived from negative review clustering.",
                target_persona="Dissatisfied users seeking better alternatives",
                core_problem="Users report dissatisfaction in product performance",
                formulation_direction="Improve reliability and performance",
                product_format="Standard product format",
                suggested_price_band="₹299 – ₹799",
                key_differentiation="Focused on solving top complaint cluster",
                why_competitors_fail="Competitors ignore recurring pain points",
                opportunity_score=min(20, int(len(g) / len(neg) * 20)),
                evidence_count=len(g),
                sample_quotes=_sample_quotes(texts),
            )
        )

    return opps[:5]


@app.get("/")
def index():
    return render_template("index.html", opportunities=None, upload_id=None, error=None)


@app.post("/analyze")
def analyze():

    file = request.files.get("file")

    if not file or not file.filename:
        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="Please upload a CSV file."
        ), 400

    try:
        df = pd.read_csv(file)

        # limit dataset for cloud deployment
        if len(df) > 500:
            df = df.sample(500, random_state=42)

    except Exception:
        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="Could not read the CSV file."
        ), 400

    cols = set([c.strip() for c in df.columns])
    missing = REQUIRED_COLUMNS - cols

    if missing:
        return (
            render_template(
                "index.html",
                opportunities=None,
                upload_id=None,
                error=f"Missing required columns: {', '.join(sorted(missing))}",
            ),
            400,
        )

    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"] = df["review_text"].map(_safe_str)

    df = df.dropna(subset=["rating", "review_text"])
    df = df[df["review_text"].str.strip().astype(bool)]

    if df.empty:
        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="CSV contains no usable rows."
        ), 400

    opportunities = _generate_opportunities(df)

    upload_id = str(uuid.uuid4())
    UPLOAD_STORE[upload_id] = opportunities

    return render_template(
        "index.html",
        opportunities=[asdict(o) for o in opportunities],
        upload_id=upload_id,
        error=None,
    )


@app.get("/plan/<upload_id>/<opp_id>")
def plan(upload_id: str, opp_id: str):

    opps = UPLOAD_STORE.get(upload_id)

    if not opps:
        return jsonify({"error": "Upload not found. Please re-upload the CSV."}), 404

    match = next((o for o in opps if o.opp_id == opp_id), None)

    if not match:
        return jsonify({"error": "Opportunity not found."}), 404

    return jsonify(asdict(match))


@app.post("/reset")
def reset():
    UPLOAD_STORE.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
