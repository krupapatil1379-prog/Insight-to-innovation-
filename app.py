import uuid
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


app = Flask(__name__)

UPLOAD_STORE = {}


# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = str(text).lower()
    return text


# ---------- CLUSTER REVIEWS ----------
def generate_opportunities(df):

    neg = df[df["rating"] <= 3].copy()

    if len(neg) == 0:
        return []

    texts = neg["review_text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform(texts)

    k = min(5, max(2, int(np.sqrt(len(texts)))))

    model = KMeans(n_clusters=k, random_state=42)

    labels = model.fit_predict(X)

    neg["cluster"] = labels

    groups = neg.groupby("cluster")

    opportunities = []

    for i, (cl, g) in enumerate(groups):

        opportunities.append({
            "opp_id": str(uuid.uuid4()),
            "theme": "Customer Pain Point",
            "product_name": f"Innovation Idea {i+1}",
            "core_problem": "Users report dissatisfaction in reviews",
            "opportunity_score": int(len(g) / len(neg) * 20),
            "evidence_count": len(g),
            "sample_quotes": g["review_text"].head(2).tolist()
        })

    return opportunities[:5]


# ---------- HOME ----------
@app.route("/")
def index():

    return render_template(
        "index.html",
        opportunities=None,
        upload_id=None,
        error=None
    )


# ---------- ANALYZE ----------
@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files.get("file")

    if not file:
        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="Upload a CSV file."
        )

    try:

        file.seek(0)

        df = pd.read_csv(file)

    except Exception:

        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="Could not read CSV."
        )

    df.columns = df.columns.str.strip().str.lower()

    if "rating" not in df.columns or "review_text" not in df.columns:

        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="CSV must contain rating and review_text columns."
        )

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df["review_text"] = df["review_text"].astype(str)

    df = df.dropna(subset=["rating", "review_text"])

    if len(df) == 0:

        return render_template(
            "index.html",
            opportunities=None,
            upload_id=None,
            error="No usable rows in CSV."
        )

    if len(df) > 200:
        df = df.head(200)

    opportunities = generate_opportunities(df)

    upload_id = str(uuid.uuid4())

    UPLOAD_STORE[upload_id] = opportunities

    return render_template(
        "index.html",
        opportunities=opportunities,
        upload_id=upload_id,
        error=None
    )


# ---------- PLAN ----------
@app.route("/plan/<upload_id>/<opp_id>")
def plan(upload_id, opp_id):

    opps = UPLOAD_STORE.get(upload_id)

    if not opps:
        return jsonify({"error": "Upload not found"}), 404

    match = next((o for o in opps if o["opp_id"] == opp_id), None)

    if not match:
        return jsonify({"error": "Opportunity not found"}), 404

    return jsonify(match)


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
