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
OPTIONAL_COLUMNS = {"product_name", "brand_name", "category"}

# Explicitly avoid showing (or even emitting) these generic clustering words in UI strings.
FORBIDDEN_OUTPUT_TERMS = {
    "hair",
    "product",
    "months",
    "used",
}

STOPWORDS_MINI = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "too",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


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
    opportunity_score: int  # 0..20
    evidence_count: int
    sample_quotes: List[str]


# Simple in-memory store of latest results per upload.
UPLOAD_STORE: Dict[str, List[Opportunity]] = {}


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _clean_text(text: str) -> str:
    text = _safe_str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sanitize_output(text: str) -> str:
    out = _safe_str(text).strip()
    if not out:
        return out

    # Remove forbidden terms as standalone words (case-insensitive).
    for term in FORBIDDEN_OUTPUT_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _pick_category_hint(df: pd.DataFrame) -> str:
    if "category" in df.columns:
        vals = df["category"].dropna().astype(str).str.strip()
        if not vals.empty:
            top = vals.value_counts().index[0]
            return top[:50]
    return ""


def _theme_from_texts(texts: List[str]) -> str:
    joined = " ".join(texts)

    # Lightweight heuristic theme detection (no keywords shown in UI).
    checks = [
        ("Skin Irritation & Sensitivity", ["itch", "rash", "irritat", "burn", "redness", "allerg"]),
        ("Dryness & Tightness", ["dry", "flak", "tight", "rough", "parch"]),
        ("Greasy Feel & Stickiness", ["greasy", "oily", "sticky", "heavy", "chip chip"]),
        ("Breakouts & Congestion", ["acne", "pimple", "breakout", "comed", "clog"]),
        ("Weak Performance / No Results", ["no effect", "doesn't work", "no result", "waste", "useless", "ineffective"]),
        ("Bad Odor / After-Smell", ["smell", "odor", "fragrance", "stink", "perfume"]),
        ("Staining / Residue", ["stain", "residue", "marks", "white cast", "build up"]),
        ("Packaging Leakage / Mess", ["leak", "spill", "mess", "broken", "pump", "cap"]),
        ("Low Durability / Breakage", ["broke", "break", "crack", "durable", "fragile"]),
        ("Battery / Charging Issues", ["battery", "charge", "charging", "drain", "power"]),
        ("Connectivity / App Issues", ["bluetooth", "connect", "app", "sync", "pair"]),
        ("Size / Fit Discomfort", ["fit", "size", "tight", "loose", "uncomfortable"]),
        ("Delivery / Damage", ["delivery", "damaged", "damage", "late", "missing"]),
    ]

    for theme, needles in checks:
        for n in needles:
            if n in joined:
                return theme

    return "Unmet Expectations & Inconsistent Experience"


def _persona_from_theme(theme: str, category_hint: str) -> str:
    base = {
        "Skin Irritation & Sensitivity": "Sensitive-skin consumers in humid Indian cities",
        "Dryness & Tightness": "Daily users seeking comfort and hydration without heaviness",
        "Greasy Feel & Stickiness": "Oily-skin users who avoid heavy-feel solutions",
        "Breakouts & Congestion": "Acne-prone consumers needing non-comedogenic performance",
        "Weak Performance / No Results": "Value-conscious buyers who demand visible results fast",
        "Bad Odor / After-Smell": "Fragrance-sensitive users who prefer clean-feel experiences",
        "Staining / Residue": "Professionals who need residue-free, camera-ready finishes",
        "Packaging Leakage / Mess": "Busy commuters who need spill-proof, travel-safe packaging",
        "Low Durability / Breakage": "Heavy users who need robust, long-lasting build quality",
        "Battery / Charging Issues": "On-the-go users who rely on predictable all-day power",
        "Connectivity / App Issues": "Digital-first users expecting seamless setup and control",
        "Size / Fit Discomfort": "All-day wearers who prioritize comfort and fit",
        "Delivery / Damage": "Online-first shoppers who expect protected, reliable delivery",
        "Unmet Expectations & Inconsistent Experience": "Mainstream consumers seeking predictable, repeatable outcomes",
    }
    persona = base.get(theme, base["Unmet Expectations & Inconsistent Experience"])
    if category_hint and len(category_hint) <= 24:
        return f"{persona} ({category_hint})"
    return persona


def _format_from_category(category_hint: str, theme: str) -> str:
    c = category_hint.lower()
    if any(x in c for x in ["skin", "beauty", "personal", "cosmetic", "face", "body", "haircare", "hair care"]):
        if "Packaging" in theme or "Staining" in theme:
            return "Pump bottle + travel mini"
        return "Serum + lightweight gel-cream duo"
    if any(x in c for x in ["home", "clean", "laundry", "dish"]):
        return "Concentrated refill pouch + dispenser"
    if any(x in c for x in ["electronics", "device", "gadget", "appliance"]):
        return "Device + companion quick-start card"
    return "Core SKU + trial-size sampler"


def _price_band(category_hint: str) -> str:
    c = category_hint.lower()
    if any(x in c for x in ["electronics", "device", "gadget", "appliance"]):
        return "₹1,999 – ₹4,999"
    if any(x in c for x in ["baby", "derma", "clinical", "pharma"]):
        return "₹399 – ₹899"
    return "₹299 – ₹799"


def _product_name(theme: str, category_hint: str, idx: int) -> str:
    # Avoid forbidden generic words and keep names brandable.
    theme_to_name = {
        "Skin Irritation & Sensitivity": "CalmShield",
        "Dryness & Tightness": "HydraEase",
        "Greasy Feel & Stickiness": "AirMatte",
        "Breakouts & Congestion": "ClearBalance",
        "Weak Performance / No Results": "ProofPulse",
        "Bad Odor / After-Smell": "FreshNeutral",
        "Staining / Residue": "ZeroTrace",
        "Packaging Leakage / Mess": "SealSure",
        "Low Durability / Breakage": "BuiltRight",
        "Battery / Charging Issues": "PowerSteady",
        "Connectivity / App Issues": "SyncSmooth",
        "Size / Fit Discomfort": "FitComfort",
        "Delivery / Damage": "PackGuard",
        "Unmet Expectations & Inconsistent Experience": "RelianceSpec",
    }
    base = theme_to_name.get(theme, "InsightForge")
    suffix = ""
    if category_hint:
        # Short, non-keyword suffix derived from category initials.
        initials = "".join([w[0].upper() for w in re.findall(r"[A-Za-z]+", category_hint)[:3]])
        if initials:
            suffix = f" {initials}"
    return _sanitize_output(f"{base}{suffix} {idx}")


def _rng_for(opp_id: str) -> np.random.Generator:
    # Deterministic per opportunity id so UI is stable across refreshes.
    h = re.sub(r"[^0-9a-fA-F]", "", opp_id)
    seed = int(h[:8], 16) if len(h) >= 8 else 42
    return np.random.default_rng(seed)


def _category_family(category_hint: str, product_format: str) -> str:
    t = f"{category_hint} {product_format}".lower()
    if any(x in t for x in ["electronics", "device", "gadget", "appliance", "bluetooth", "app"]):
        return "electronics"
    if any(x in t for x in ["skin", "beauty", "cosmetic", "face", "body", "derma", "serum", "cream"]):
        return "personal_care"
    if any(x in t for x in ["home", "clean", "laundry", "dish", "refill", "dispenser"]):
        return "home_care"
    return "general"


def _dynamic_roadmap(opportunity: Opportunity) -> List[Tuple[str, str]]:
    rng = _rng_for(opportunity.opp_id)
    fam = _category_family(opportunity.category_hint, opportunity.product_format)
    theme = opportunity.theme
    name = opportunity.product_name
    persona_short = opportunity.target_persona.split("(")[0].strip()
    diff_short = opportunity.key_differentiation.split(".")[0].strip()

    # Pick a roadmap template per family/theme; each phase is on separate lines in UI.
    if fam == "electronics" or theme in {"Battery / Charging Issues", "Connectivity / App Issues"}:
        phase_1 = rng.choice(
            [
                "Hardware spec + firmware baseline; validate key failure modes and battery/pairing stability.",
                "Define reliability spec + UX flows; build first engineering prototype and log real-world performance.",
                "Fix top reliability issues; lock core user journeys and establish pass/fail test gates.",
            ]
        )
        phase_2 = rng.choice(
            [
                "Closed beta with target users; iterate pairing/onboarding and stabilize power management.",
                "Pilot with 50–100 users; prioritize crash-free sessions and reduce connection drops.",
                "Field test across handset mix; refine app fallbacks and improve charging predictability.",
            ]
        )
        phase_3 = rng.choice(
            [
                "Launch proof assets: reliability benchmarks, battery life screenshots, and setup-in-60s demos.",
                "Creator-led demos focused on frictionless setup; publish stability proof points and FAQs.",
                "Performance launch with ‘no-fuss setup’ messaging; show real-day usage and fast support path.",
            ]
        )
        phase_4 = rng.choice(
            [
                "Marketplace listings + service readiness; ship updated firmware and in-box quick-start.",
                "Channel onboarding + warranty positioning; optimize PDP with troubleshooting clarity.",
                "Scale distribution; strengthen after-sales and first-30-day success program.",
            ]
        )
        phase_5 = rng.choice(
            [
                "Bundle offers + accessory attach; drive reviews and reduce returns with proactive support.",
                "Referral loop via early adopters; focus on retention and firmware updates.",
                "Targeted remarketing to high-intent buyers; improve review velocity and response times.",
            ]
        )
        phase_6 = rng.choice(
            [
                "Scale performance marketing; expand cities and optimize CAC with reliability proof.",
                "Iterate product based on telemetry; launch v1.1 update and widen retail presence.",
                "Grow via partnerships and seasonal campaigns; push NPS and repeat usage metrics.",
            ]
        )
        return [
            ("Day 1–15", phase_1),
            ("Day 16–30", phase_2),
            ("Day 31–45", phase_3),
            ("Day 46–60", phase_4),
            ("Day 61–75", phase_5),
            ("Day 76–90", phase_6),
        ]

    if theme in {"Packaging Leakage / Mess", "Delivery / Damage"}:
        phase_1 = rng.choice(
            [
                "Redesign primary packaging; run leak/drop/heat-cycle tests for Indian logistics.",
                "Lock packaging spec + sealing method; prototype and validate travel + courier handling.",
                "Identify top failure points; create v1 packaging and stress-test under heat and drop scenarios.",
            ]
        )
        phase_2 = rng.choice(
            [
                "Pilot with courier partners; iterate protective packing and sealing QC checks.",
                "Small-batch trial; monitor leakage/damage rate and refine packaging tolerances.",
                "Beta ship to employees/early adopters; capture unboxing feedback and failure analytics.",
            ]
        )
        phase_3 = rng.choice(
            [
                "Launch with ‘zero-mess’ proof: drop tests, travel demos, and tamper-evident highlights.",
                "Creator unboxing + travel tests; publish durability proof and clear replacement policy.",
                "Performance launch with packaging proof content; emphasize reliability and waste reduction.",
            ]
        )
        phase_4 = rng.choice(
            [
                "Marketplace rollout; optimize packing SOPs and tighten vendor QC sampling.",
                "Scale shipments; implement damage prevention checklist and faster replacement flow.",
                "Expand channels; improve warehouse handling and monitor returns by lane.",
            ]
        )
        phase_5 = rng.choice(
            [
                "Bundle travel minis; drive repeat purchase with ‘carry-anywhere’ positioning.",
                "Retention offers for reorders; use packaging reliability as the hero proof point.",
                "Referral loop: ‘mess-free guarantee’ campaign to boost advocacy and reviews.",
            ]
        )
        phase_6 = rng.choice(
            [
                "Scale marketing; expand distribution and measure leakage/damage rate week-over-week.",
                "Optimize CAC using proof creatives; expand variants and improve unit economics.",
                "Scale nationally with logistics scorecards; maintain QC gates as volumes grow.",
            ]
        )
        return [
            ("Day 1–15", phase_1),
            ("Day 16–30", phase_2),
            ("Day 31–45", phase_3),
            ("Day 46–60", phase_4),
            ("Day 61–75", phase_5),
            ("Day 76–90", phase_6),
        ]

    # Personal/home care: make roadmap truly concept-specific (not generic).
    theme_hooks = {
        "Skin Irritation & Sensitivity": {
            "demo": "‘Comfort Stress-Test’ (heat + sweat) showing zero-sting feel",
            "proof": "barrier-support proof points and irritation-minimization guardrails",
            "community": "derm-led education + sensitive-skin micro-communities",
        },
        "Dryness & Tightness": {
            "demo": "‘8-Hour Comfort Check’ hydration hold without heaviness",
            "proof": "hydration retention + comfort scoring in humid conditions",
            "community": "working professionals + AC-to-outdoor routine creators",
        },
        "Greasy Feel & Stickiness": {
            "demo": "‘Humidity Matte Challenge’ (midday reapply, no sticky feel)",
            "proof": "texture-to-finish proof (fast absorption + breathable matte)",
            "community": "oily-skin creators + commute/workday proof content",
        },
        "Breakouts & Congestion": {
            "demo": "‘No-Backfire Routine’ weekly diary (results without congestion)",
            "proof": "non-clogging base + simplified irritant profile proof points",
            "community": "acne-prone routine communities + pharmacist/derm explainers",
        },
        "Bad Odor / After-Smell": {
            "demo": "‘Close-Contact Test’ (all-day fresh-feel, no after-smell)",
            "proof": "odor-neutral tech + low-residual fragrance system",
            "community": "gym commuters + office wearers who care about after-smell",
        },
        "Staining / Residue": {
            "demo": "‘White-Shirt Proof’ + camera-ready finish in harsh lighting",
            "proof": "residue-free film + no transfer performance proof",
            "community": "professionals + creators focused on wardrobe-safe routines",
        },
        "Weak Performance / No Results": {
            "demo": "‘Visible Wins in 7 Days’ proof series with guardrailed claims",
            "proof": "clear measurement standard: what changes and when",
            "community": "value-conscious shoppers + comparison content",
        },
        "Unmet Expectations & Inconsistent Experience": {
            "demo": "‘Consistency Guarantee’ (same outcome, every time) demo",
            "proof": "tight spec + fewer failure modes messaging",
            "community": "mainstream daily users + reliability-focused proof content",
        },
    }
    hook = theme_hooks.get(theme, theme_hooks["Unmet Expectations & Inconsistent Experience"])

    channels = rng.choice(
        [
            "Nykaa + brand website",
            "Amazon + brand website",
            "Nykaa + quick-commerce pilots (Blinkit/Zepto)",
            "D2C-first with creator-led demand, then marketplaces",
        ]
    )
    content_device = rng.choice(
        [
            "a 10-day creator diary",
            "a side-by-side comparison series",
            "a ‘myth vs proof’ explainer set",
            "short routine reels + a long-form deep-dive",
        ]
    )
    sampling = rng.choice(
        [
            "50 micro-influencers in 3 metro cities",
            "25 derm/skin-science creators + 30 routine creators",
            "a campus + office sampling wave across 2 cities",
            "community seeding to 100 early adopters via waitlist",
        ]
    )

    phase_1 = _sanitize_output(
        f"Lock {name} prototype(s) around: {diff_short}; define pass/fail checks for Indian heat/humidity and daily reapplication."
    )
    phase_2 = _sanitize_output(
        f"Run beta with {sampling}; collect structured feedback from {persona_short} and iterate on the top 2 friction points."
    )
    phase_3 = _sanitize_output(
        f"Launch on {channels} using {hook['demo']} via {content_device}; publish proof assets tied to {hook['proof']}."
    )
    phase_4 = _sanitize_output(
        f"Scale with marketplace PDP upgrades (FAQs, proof, usage guidance); introduce trial kit + bundle to reduce hesitation and improve reviews."
    )
    phase_5 = _sanitize_output(
        f"Build repeat purchase by activating {hook['community']}; retarget by pain-point segments and double down on the best-performing pillar."
    )
    phase_6 = _sanitize_output(
        f"Expand nationally with performance marketing and retention loops (CRM + repurchase reminders); optimize CAC using the strongest proof creative."
    )
    return [
        ("Day 1–15", phase_1),
        ("Day 16–30", phase_2),
        ("Day 31–45", phase_3),
        ("Day 46–60", phase_4),
        ("Day 61–75", phase_5),
        ("Day 76–90", phase_6),
    ]


def _dynamic_messaging(opportunity: Opportunity) -> Tuple[List[str], List[str]]:
    rng = _rng_for(opportunity.opp_id)
    fam = _category_family(opportunity.category_hint, opportunity.product_format)
    theme = opportunity.theme

    persona_short = opportunity.target_persona.split("(")[0].strip()
    diff_short = opportunity.key_differentiation.split(".")[0].strip()
    problem_short = opportunity.core_problem.strip().rstrip(".")

    pillar_pool = {
        "electronics": [
            "Reliability You Can Trust, Every Day",
            "Setup That Works in Under a Minute",
            "Proof-Backed Performance (Not Just Specs)",
            "Built for Indian Power & Network Conditions",
            "Fewer Failure Modes, More Real-Life Wins",
        ],
        "personal_care": [
            "Comfort-First Performance in Humidity",
            "Barrier/Balance-Friendly by Design",
            "Fast Absorption, Clean Feel",
            "Results Without the Trade-offs",
            "Routine-Ready for Indian Weather",
        ],
        "home_care": [
            "Mess-Free, Refill-Ready Convenience",
            "High Performance With Less Waste",
            "Designed for Indian Kitchens & Homes",
            "Consistent Results, Every Use",
            "Easy to Store, Easy to Reorder",
        ],
        "general": [
            "Reliability-First, Everyday Performance",
            "Designed for Real Indian Routines",
            "Proof That’s Easy to Understand",
            "Lower Friction, Higher Satisfaction",
            "Consistency You Can Feel",
        ],
    }

    headline_pool = {
        "electronics": [
            "“Pair Once. It Stays Paired.”",
            "“All-Day Power, Without the Anxiety.”",
            "“Setup That Doesn’t Need a Tutorial.”",
            "“Built for Real Indian Days.”",
            "“Performance You Can Measure.”",
        ],
        "personal_care": [
            "“Results Without the Sting.”",
            "“Hydration That Doesn’t Turn Heavy.”",
            "“Matte Feel. Real Performance.”",
            "“Built for Humidity, Not Just Labels.”",
            "“Comfort That Lasts.”",
        ],
        "home_care": [
            "“No Mess. No Waste. Just Results.”",
            "“Refill, Reuse, Repeat—Better.”",
            "“Powerful Clean, Cleaner Routine.”",
            "“Built for Everyday Indian Homes.”",
            "“Consistent From First Use to Last.”",
        ],
        "general": [
            "“Stop Settling for Inconsistency.”",
            "“Built for Real Life, Not Best Case.”",
            "“Performance You Can Trust.”",
            "“Less Friction. More Results.”",
            "“Designed for Indian Conditions.”",
        ],
    }

    # Theme-specific spice to ensure distinctiveness.
    theme_spice = {
        "Skin Irritation & Sensitivity": ("Comfort-First Strength", "“Strong Results, Gentle Experience.”"),
        "Dryness & Tightness": ("Long-Wear Comfort Hydration", "“Hydration That Stays Comfortable.”"),
        "Greasy Feel & Stickiness": ("Weightless Finish Technology", "“Matte Feel, All-Day Confidence.”"),
        "Breakouts & Congestion": ("Non-Clogging Performance", "“Clearer Results Without the Backfire.”"),
        "Packaging Leakage / Mess": ("Zero-Mess Reliability", "“Carry It. Ship It. Trust It.”"),
        "Battery / Charging Issues": ("Predictable Power", "“Charge Fast. Last Longer.”"),
        "Connectivity / App Issues": ("Frictionless Connection", "“It Just Works—Every Time.”"),
    }.get(theme, ("Consistency Engineered", "“Predictable, Repeatable Results.”"))

    pillars = list(rng.choice(pillar_pool.get(fam, pillar_pool["general"]), size=2, replace=False))
    pillars.append(f"{theme_spice[0]} for {persona_short}")
    pillars = [_sanitize_output(p) for p in pillars][:3]

    # Build 3 headlines: 1 from pool, 1 theme-specific, 1 product-specific.
    pool_pick = str(rng.choice(headline_pool.get(fam, headline_pool["general"])))
    product_specific = f"“{opportunity.product_name}: {diff_short}.”"
    problem_specific = f"“Fix: {problem_short}.”"
    candidates = [pool_pick, theme_spice[1], product_specific, problem_specific]
    rng.shuffle(candidates)
    headlines = []
    for h in candidates:
        h2 = _sanitize_output(h.strip("“”").strip())
        if not h2:
            continue
        # Keep headlines unique.
        if h2.lower() in {x.lower() for x in headlines}:
            continue
        headlines.append(h2)
        if len(headlines) >= 3:
            break

    return pillars, headlines


def _concept_from_theme(theme: str, category_hint: str) -> Tuple[str, str, str, str, str]:
    # Returns: product_concept, core_problem, formulation_direction, differentiation, competitor_fail
    if theme == "Skin Irritation & Sensitivity":
        return (
            "A comfort-first, high-performance solution engineered for Indian heat, sweat, and sensitive skin barriers.",
            "Consumers experience stinging, redness, or discomfort during regular use—especially in humid conditions.",
            "Barrier-support complex (ceramides + panthenol) with fragrance-minimized system and sweat-tolerant texture.",
            "Clinical-strength results with a comfort-first sensory profile for Indian climate and sensitive users.",
            "Competitors chase strong actives but ignore barrier support and climate-driven reactivity, causing irritation and drop-off.",
        )
    if theme == "Dryness & Tightness":
        return (
            "A lightweight, long-wear hydration concept that stays comfortable in heat without feeling heavy.",
            "Users report tight, uncomfortable feel and visible dryness shortly after application or repeated use.",
            "Multi-weight humectants + occlusion-lite film former to hold hydration without greasy feel.",
            "Comfort that lasts in humidity—hydration without shine or heaviness.",
            "Competitors over-occlude (feels heavy) or under-deliver (hydration fades fast), leading to inconsistent satisfaction.",
        )
    if theme == "Greasy Feel & Stickiness":
        return (
            "A fast-absorbing, matte-finish performance concept optimized for Indian humidity and daily reapplication.",
            "Consumers avoid repeat use because the feel is sticky or heavy, especially during warm weather.",
            "Fast-absorption emollients + powder-microgel network for breathable matte finish.",
            "High performance with a weightless sensory profile that survives Indian humidity.",
            "Competitors focus on claims but neglect sensory engineering; sticky feel kills repeat purchase.",
        )
    if theme == "Breakouts & Congestion":
        return (
            "A non-clogging, balance-first concept that delivers results while respecting acne-prone routines.",
            "Users report congestion or breakouts after using the solution consistently.",
            "Non-comedogenic base with microbiome-friendly actives and simplified irritant profile.",
            "Results without triggering congestion—built for acne-prone Indian consumers.",
            "Competitors overload heavy bases or irritants; outcomes improve on paper but worsen real-life routines.",
        )
    if theme == "Packaging Leakage / Mess":
        return (
            "A leak-proof, travel-safe concept designed for Indian commuting and e-commerce handling.",
            "Users experience leaks, spills, or broken components leading to waste and frustration.",
            "Torque-locked closures, gasket sealing, and drop-tested primary packaging.",
            "Reliability-first packaging that protects experience from delivery to daily carry.",
            "Competitors optimize cost; weak seals and brittle components create leakage and negative word of mouth.",
        )
    if theme == "Battery / Charging Issues":
        return (
            "A predictable all-day power concept with clear status communication and quick-charge reliability.",
            "Consumers report rapid drain, unreliable charging, or unclear power status.",
            "Battery health management + quick-charge circuitry + clearer low-power signaling.",
            "Trustable power you can plan around—built for all-day Indian usage patterns.",
            "Competitors chase specs but ignore real-world power management and user feedback cues, causing churn.",
        )
    if theme == "Connectivity / App Issues":
        return (
            "A frictionless setup and control experience that works reliably across common Indian handset conditions.",
            "Consumers face pairing drops, setup failures, or unpredictable app behavior.",
            "Stronger pairing flow, offline-first controls, and simplified onboarding with robust fallbacks.",
            "Smooth connection that ‘just works’—even on variable networks and device mixes.",
            "Competitors build feature-heavy apps without reliability engineering, creating daily frustration.",
        )

    # Default
    return (
        "A reliability-first concept that standardizes outcomes and removes common friction points in day-to-day use.",
        "Consumers report inconsistent performance and a gap between promise and lived experience.",
        "Tightened spec, fewer failure modes, and a performance system tuned for Indian conditions.",
        "Predictable outcomes with fewer trade-offs—engineered for repeatable everyday performance.",
        "Competitors overpromise and under-engineer consistency, leading to disappointment and negative reviews.",
    )


def _sample_quotes(texts: List[str], k: int = 2) -> List[str]:
    # Keep quotes short and safe; do not expose cluster keywords as a list.
    out: List[str] = []
    for t in texts[: 10 * k]:
        s = _sanitize_output(_safe_str(t))
        if not s:
            continue
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > 140:
            s = s[:137].rstrip() + "..."
        out.append(s)
        if len(out) >= k:
            break
    return out


def _opportunity_score(cluster_size: int, avg_rating: float, unique_users_proxy: int) -> int:
    # Simple 20-point score: prevalence (0..10) + severity (0..10)
    prevalence = min(10.0, 10.0 * (cluster_size / max(1, unique_users_proxy)))
    severity = min(10.0, 10.0 * ((3.0 - avg_rating) / 2.0))  # rating 1..3 -> 10..0-ish
    score = int(round(prevalence + severity))
    return max(0, min(20, score))


def _cluster_negative_reviews(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Optional[KMeans]]:
    neg = df[df["rating"] <= 3].copy()

    # limit dataset for render free server
    if len(neg) > 200:
        neg = neg.sample(200, random_state=42)

    neg["clean_text"] = neg["review_text"].map(clean_text)
    neg = neg[neg["clean_text"].str.len() >= 8].copy()

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

    # Choose k to ensure we can always return exactly 5 opportunities after ranking.
    # We can rank clusters and then map them into exactly 5 concepts even if k < 5.
    k = int(np.clip(round(np.sqrt(max(2, len(docs) / 2))), 2, 8))
    if len(docs) < 10:
        k = 2
    if len(docs) < 4:
        k = 1

    # Use an explicit int for broad scikit-learn compatibility.
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    neg["cluster"] = labels
    return neg, labels, model


def _generate_opportunities(df: pd.DataFrame) -> List[Opportunity]:
    neg, labels, _ = _cluster_negative_reviews(df)
    if neg.empty:
        # No negative reviews: still generate 5 opportunities, but mark low evidence.
        cat = _pick_category_hint(df)
        base_theme = "Unmet Expectations & Inconsistent Experience"
        opps: List[Opportunity] = []
        for i in range(1, 6):
            theme = base_theme
            persona = _persona_from_theme(theme, cat)
            concept, problem, formulation, diff, fail = _concept_from_theme(theme, cat)
            opps.append(
                Opportunity(
                    opp_id=str(uuid.uuid4()),
                    theme=theme,
                    category_hint=cat,
                    product_name=_product_name(theme, cat, i),
                    product_concept=_sanitize_output(concept),
                    target_persona=_sanitize_output(persona),
                    core_problem=_sanitize_output(problem),
                    formulation_direction=_sanitize_output(formulation),
                    product_format=_sanitize_output(_format_from_category(cat, theme)),
                    suggested_price_band=_sanitize_output(_price_band(cat)),
                    key_differentiation=_sanitize_output(diff),
                    why_competitors_fail=_sanitize_output(fail),
                    opportunity_score=6,
                    evidence_count=0,
                    sample_quotes=[],
                )
            )
        return opps

    cat = _pick_category_hint(df)
    cluster_groups = neg.groupby("cluster")
    total_neg = int(len(neg))

    ranked: List[Tuple[int, int]] = []
    for cl, g in cluster_groups:
        avg_rating = float(g["rating"].mean()) if not g.empty else 3.0
        score = _opportunity_score(cluster_size=len(g), avg_rating=avg_rating, unique_users_proxy=total_neg)
        ranked.append((int(cl), int(score)))

    ranked.sort(key=lambda x: (x[1], len(cluster_groups.get_group(x[0]))), reverse=True)

    # Create up to 8 raw concepts, then compress into exactly 5 opportunities.
    raw_concepts: List[Opportunity] = []
    for idx, (cl, score) in enumerate(ranked[:8], start=1):
        g = cluster_groups.get_group(cl).copy()
        texts = g["review_text"].astype(str).tolist()
        clean_texts = g["clean_text"].astype(str).tolist()
        theme = _theme_from_texts(clean_texts)
        persona = _persona_from_theme(theme, cat)
        concept, problem, formulation, diff, fail = _concept_from_theme(theme, cat)
        raw_concepts.append(
            Opportunity(
                opp_id=str(uuid.uuid4()),
                theme=theme,
                category_hint=cat,
                product_name=_product_name(theme, cat, idx),
                product_concept=_sanitize_output(concept),
                target_persona=_sanitize_output(persona),
                core_problem=_sanitize_output(problem),
                formulation_direction=_sanitize_output(formulation),
                product_format=_sanitize_output(_format_from_category(cat, theme)),
                suggested_price_band=_sanitize_output(_price_band(cat)),
                key_differentiation=_sanitize_output(diff),
                why_competitors_fail=_sanitize_output(fail),
                opportunity_score=int(np.clip(score, 0, 20)),
                evidence_count=int(len(g)),
                sample_quotes=_sample_quotes(texts, k=2),
            )
        )

    # Ensure exactly 5 opportunities, with unique messaging anchors.
    opps: List[Opportunity] = []
    used_themes: set = set()
    for c in raw_concepts:
        # Reduce duplicates by theme.
        theme_key = _theme_from_texts([_clean_text(c.core_problem)])
        if theme_key in used_themes and len(opps) < 5:
            continue
        used_themes.add(theme_key)
        opps.append(c)
        if len(opps) >= 5:
            break

    # If still < 5 (small dataset), pad using best available with slight variations.
    i = 1
    while len(opps) < 5:
        seed = raw_concepts[min(len(raw_concepts) - 1, 0)]
        theme = "Unmet Expectations & Inconsistent Experience"
        persona = _persona_from_theme(theme, cat)
        concept, problem, formulation, diff, fail = _concept_from_theme(theme, cat)
        opps.append(
            Opportunity(
                opp_id=str(uuid.uuid4()),
                theme=theme,
                category_hint=cat,
                product_name=_product_name(theme, cat, 100 + len(opps) + 1),
                product_concept=_sanitize_output(concept),
                target_persona=_sanitize_output(persona),
                core_problem=_sanitize_output(problem),
                formulation_direction=_sanitize_output(formulation),
                product_format=_sanitize_output(_format_from_category(cat, theme)),
                suggested_price_band=_sanitize_output(_price_band(cat)),
                key_differentiation=_sanitize_output(diff),
                why_competitors_fail=_sanitize_output(fail),
                opportunity_score=max(5, min(12, seed.opportunity_score - i)),
                evidence_count=0,
                sample_quotes=[],
            )
        )
        i += 1

    # Hard guarantee.
    return opps[:5]


def _plan_for(opportunity: Opportunity) -> Dict:
    score = int(np.clip(opportunity.opportunity_score, 0, 20))

    win_india = (
        "India’s climate, long commutes, and high price-sensitivity punish solutions that feel uncomfortable, fail in humidity, "
        "or degrade after delivery. A reliability-first concept that’s engineered for Indian conditions—and communicated with proof—"
        "can earn trust, repeat purchase, and advocacy."
    )

    executive = (
        f"Negative reviews consistently point to a repeatable pain point: {opportunity.core_problem} "
        f"That makes this a high-leverage opportunity for a concept that delivers {opportunity.key_differentiation.lower()}."
    )

    positioning = (
        f"For {opportunity.target_persona}, {opportunity.product_name} is the solution that solves "
        f"'{opportunity.core_problem}' with {opportunity.key_differentiation.lower()}, unlike typical options that "
        f"{opportunity.why_competitors_fail.lower()}."
    )

    roadmap = _dynamic_roadmap(opportunity)
    pillars, headlines = _dynamic_messaging(opportunity)

    campaign_angles = [
        "“Real Routine, Real Proof” — creators demonstrate across a full Indian day (commute → work → evening).",
        "“Heat-Test Challenge” — show performance under Indian humidity with before/after proof moments.",
        "“No-Compromise Week” — users document daily use and what improves without the usual trade-offs.",
        "“Reliability Guarantee” — highlight the brand’s quality gates and customer-first replacement promise.",
    ]
    rng = _rng_for(opportunity.opp_id)
    angle = str(rng.choice(campaign_angles))
    campaign = (
        f"{angle} The story anchors on the core problem ({opportunity.core_problem}) and why {opportunity.product_name} "
        f"wins with {opportunity.key_differentiation.lower()}."
    )

    fam = _category_family(opportunity.category_hint, opportunity.product_format)
    base_kpis = [
        "PDP conversion rate and add-to-cart lift",
        "Review sentiment shift on core-problem mentions",
        "Creator content completion rate and saves/shares",
    ]
    if fam == "electronics":
        extra = ["Return rate and top return reasons", "Activation success rate (setup completed)", "7-day active usage rate"]
    elif opportunity.theme in {"Packaging Leakage / Mess", "Delivery / Damage"}:
        extra = ["Damage/leakage rate per 1,000 shipments", "Replacement request rate", "Repeat purchase rate"]
    else:
        extra = ["Repeat purchase intent (survey) and 30-day repurchase rate", "Return rate and top return reasons", "Trial-to-full conversion rate"]
    kpis = base_kpis + extra[:2]

    risks = [
        ("Claims Overreach", "Keep claims proof-backed; prioritize measurable, defensible outcomes."),
        ("Execution Drift", "Lock a spec checklist; review early batches weekly against the non-negotiables."),
        ("Copycat Competition", "Defend with proprietary process, proof assets, and fast iteration cadence."),
    ]
    if fam == "electronics":
        risks.insert(1, ("Reliability Regressions", "Implement test gates for pairing/power; ship OTA updates and monitor telemetry."))
    elif opportunity.theme in {"Packaging Leakage / Mess", "Delivery / Damage"}:
        risks.insert(1, ("Logistics Breakage", "Drop/heat-cycle test and tighten vendor QC; add protective packing SOPs."))
    else:
        risks.insert(1, ("Sensory Mismatch", "Run multi-city sensory testing (humid + dry regions) before scale."))

    confidence = "High" if score >= 15 else ("Medium" if score >= 10 else "Cautious")

    return {
        "Opportunity Score (X / 20)": f"{score} / 20",
        "Why This Opportunity Could Win in India": _sanitize_output(win_india),
        "Executive Insight": _sanitize_output(executive),
        "Positioning Statement": _sanitize_output(positioning),
        "90-Day Go-To-Market Roadmap": [{"phase": p, "plan": t} for p, t in roadmap],
        "Messaging Architecture": {
            "Pillars": pillars,
            "Example Headlines": headlines,
        },
        "Launch Campaign Concept": _sanitize_output(campaign),
        "KPI Framework": kpis,
        "Risk Factors & Mitigation Strategy": [{"Risk": r, "Mitigation": m} for r, m in risks],
        "Strategic Confidence Level": confidence,
    }


@app.get("/")
def index():
    return render_template("index.html", opportunities=None, upload_id=None, error=None)


@app.post("/analyze")
def analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        return render_template("index.html", opportunities=None, upload_id=None, error="Please upload a CSV file."), 400

    try:
        df = pd.read_csv(file)

# limit dataset for cloud deployment
if len(df) > 500:
    df = df.sample(500, random_state=42)
    except Exception:
        return render_template("index.html", opportunities=None, upload_id=None, error="Could not read the CSV file."), 400

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

    # Normalize required columns.
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"] = df["review_text"].map(_safe_str)
    df = df.dropna(subset=["rating", "review_text"])
    df = df[df["review_text"].str.strip().astype(bool)]

    if df.empty:
        return render_template("index.html", opportunities=None, upload_id=None, error="CSV contains no usable rows."), 400

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

    return jsonify(_plan_for(match))


@app.post("/reset")
def reset():
    UPLOAD_STORE.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)

