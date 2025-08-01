"""
Standalone Streamlit web application for LogiFi.

This script rebuilds the LogiFi prototype from scratch based on the
mind‑mapping diagrams. It uses Python and Streamlit to create a
bilingual (English/Arabic) website that follows the "before → after"
customer journey and incorporates key features such as data upload,
risk analysis (with an interactive scenario simulator and PDF report),
and a simple chat assistant. Colours and styling mirror the existing
brand (dark background with gold accents).

To run the app locally:

    streamlit run new_site.py

Note: This script does not require external APIs and does not persist
user data beyond the current session.
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import base64
import sqlite3

# Additional imports for ML and PDF generation
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import joblib
from functools import lru_cache


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def calculate_risk(params: dict) -> float:
    """Calculate or predict a basic risk score from supply‑chain parameters.

    Attempts to use a trained machine‑learning model if available.  If
    the model cannot be loaded or prediction fails, falls back to a
    heuristic formula that increases risk with inventory value and delay
    days, adds proportional cost for FX and commodity exposure, and
    slightly reduces risk for higher supplier count.

    Args:
        params: Dictionary containing the keys ``inventory_value``,
            ``monthly_revenue``, ``num_suppliers``, ``delay_days``,
            ``fx_exposure`` and ``commodity_dependence``.
    Returns:
        A float representing the estimated risk.
    """
    # Extract features
    inventory_value = params.get("inventory_value", 0.0)
    monthly_revenue = params.get("monthly_revenue", 1.0)
    num_suppliers = params.get("num_suppliers", 1)
    delay_days = params.get("delay_days", 0)
    fx_exposure = params.get("fx_exposure", 0)
    commodity_dependence = params.get("commodity_dependence", 0)

    # Attempt to use the trained model
    model = _load_ml_model()
    if model is not None:
        try:
            features = [
                inventory_value,
                monthly_revenue,
                num_suppliers,
                delay_days,
                fx_exposure,
                commodity_dependence,
            ]
            prediction = model.predict([features])[0]
            return float(prediction)
        except Exception:
            # fall back to heuristic
            pass

    # Heuristic fallback
    base_risk = (inventory_value * delay_days) / max(monthly_revenue, 1)
    fx_risk = (fx_exposure / 100) * monthly_revenue * 0.3
    commodity_risk = (commodity_dependence / 100) * inventory_value * 0.2
    supplier_factor = (num_suppliers ** -0.5) * 50000
    return base_risk + fx_risk + commodity_risk + supplier_factor


# Load trained model once per session
@lru_cache(maxsize=1)
def _load_ml_model(path: str = "risk_model.pkl"):
    """Load the trained RandomForest risk model from disk.

    Uses LRU caching so the model is only loaded once during a session.  If
    the model file is missing or invalid, returns ``None``.
    """
    try:
        model = joblib.load(path)
        return model
    except Exception:
        return None


def analyze_question(question: str, df: pd.DataFrame, language: str = "English", mode: str = "Beginner") -> str:
    """Enhanced Q&A handler based on the user's risk data.

    This function answers common supply‑chain risk queries and supports a
    glossary mode for explaining financial terms.  It also honours a
    beginner/pro mode by returning more forgiving answers when the user is
    unfamiliar with financial terminology.  For threshold queries, the
    question may contain a number (e.g. "above 50000") and keywords
    indicating which column to filter on.  Glossary queries starting
    with "what is" (English) or "ما هو" (Arabic) return simple
    definitions and examples based on the uploaded data.

    Args:
        question: The user's natural‑language query.
        df: DataFrame with a ``predicted_risk`` column (if absent, it
            will be computed on the fly).
        language: Either "English" or "العربية".
        mode: Either "Beginner" or "Pro" to adjust responses.

    Returns:
        A string answer in the requested language.
    """
    import re
    from difflib import get_close_matches

    # Ensure predicted_risk exists for threshold calculations and definitions
    if "predicted_risk" not in df.columns:
        try:
            df["predicted_risk"] = df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
        except Exception:
            # if prediction fails, assume zero risk
            df["predicted_risk"] = 0

    q_raw = question.strip()
    q = q_raw.lower()
    numbers = re.findall(r"\d+", q)
    threshold = float(numbers[0]) if numbers else None

    # Definitions / glossary terms for English and Arabic
    # When updating definitions, include examples from df where possible.
    avg_pred_risk = float(df["predicted_risk"].mean()) if len(df) else 0.0
    avg_fx = float(df["fx_exposure"].mean()) if "fx_exposure" in df.columns else 0.0
    avg_comm = float(df["commodity_dependence"].mean()) if "commodity_dependence" in df.columns else 0.0
    definitions_en = {
        "risk": f"Risk measures potential financial loss or disruption. In this context, predicted risk is calculated from your data; your average predicted risk is about {avg_pred_risk:,.0f}.",
        "predicted risk": f"Predicted risk is an estimate of financial loss based on factors like inventory value, FX exposure and delays. Your current average predicted risk is {avg_pred_risk:,.0f}.",
        "fx exposure": f"FX exposure refers to sensitivity to currency exchange rate fluctuations. Your average FX exposure is about {avg_fx:.0f}%.",
        "currency exposure": f"FX exposure refers to sensitivity to currency exchange rate fluctuations. Your average FX exposure is about {avg_fx:.0f}%.",
        "commodity dependence": f"Commodity dependence represents reliance on commodity prices; higher dependence increases vulnerability to price spikes. Your average commodity dependence is about {avg_comm:.0f}%.",
        "inventory value": "Inventory value is the monetary value of your stored goods; high values increase risk if delays occur.",
        "delay days": "Delay days indicate average shipping or production delays for suppliers; longer delays heighten risk.",
    }
    definitions_ar = {
        "المخاطر": f"المخاطر تقيس احتمال الخسارة المالية أو التعطل. في هذا السياق، يتم حساب المخاطر المتوقعة من بياناتك؛ متوسط المخاطر المتوقعة يبلغ حوالي {avg_pred_risk:,.0f}.",
        "المخاطر المتوقعة": f"المخاطر المتوقعة هي تقدير للخطر المالي بناءً على عوامل مثل قيمة المخزون والتعرض للعملة والتأخيرات. متوسط المخاطر المتوقعة لديك هو {avg_pred_risk:,.0f}.",
        "التعرض للعملة": f"التعرض للعملة يعني حساسية سلسلة التوريد لتغيرات سعر الصرف. متوسط التعرض للعملة لديك هو حوالي {avg_fx:.0f}%.",
        "تعرض العملة": f"التعرض للعملة يعني حساسية سلسلة التوريد لتغيرات سعر الصرف. متوسط التعرض للعملة لديك هو حوالي {avg_fx:.0f}%.",
        "الاعتماد على السلع": f"الاعتماد على السلع يشير إلى مدى اعتماد مورديك على أسعار السلع؛ ارتفاع الاعتماد يزيد من التعرض للتقلبات. متوسط الاعتماد على السلع لديك هو حوالي {avg_comm:.0f}%.",
        "قيمة المخزون": "قيمة المخزون هي القيمة المالية للبضائع المخزنة؛ القيم العالية تزيد المخاطر إذا حدثت تأخيرات.",
        "أيام التأخير": "أيام التأخير تمثل متوسط التأخير في الشحن أو الإنتاج للموردين؛ التأخيرات الطويلة تزيد المخاطر.",
    }

    # Glossary detection
    # English: queries like "what is risk" or "define risk"
    # Arabic: queries like "ما هو المخاطر" or "ما هي المخاطر"
    if language == "English":
        # look for "what is" or "define"
        if q.startswith("what is ") or q.startswith("what is?") or q.startswith("define "):
            # extract term after the phrase
            term = q.replace("what is", "").replace("define", "").strip().strip("?")
            # match approximate key
            key = get_close_matches(term, definitions_en.keys(), n=1)
            if key:
                return definitions_en[key[0]]
    else:
        # Arabic glossary patterns
        # remove punctuation and prefixes
        q_ar = q.replace("؟", "").strip()
        if q_ar.startswith("ما هو ") or q_ar.startswith("ماهي ") or q_ar.startswith("ما هي "):
            term = q_ar.split(" ", 2)[-1].strip()
            key = get_close_matches(term, definitions_ar.keys(), n=1)
            if key:
                return definitions_ar[key[0]]

    # Keywords for numeric queries (English + Arabic)
    risk_kw = ["risk", "مخاطر"]
    delay_kw = ["delay", "تأخير"]
    fx_kw = ["fx", "currency", "عملة"]
    commodity_kw = ["commodity", "سلع", "الاعتماد على السلع", "اعتماد"]

    # Highest risk / max
    if any(word in q for word in ["highest", "max", "أعلى", "أكبر"]):
        idx = df["predicted_risk"].idxmax()
        row = df.loc[idx]
        name = row.get("supplier_name", f"Supplier {idx+1}")
        risk_val = row["predicted_risk"]
        return (
            f"The highest risk is with {name} ({risk_val:,.0f})"
            if language == "English" else
            f"أعلى مستوى للمخاطر هو لدى {name} ({risk_val:,.0f})"
        )

    # Average risk
    if any(word in q for word in ["average", "mean", "متوسط"]):
        avg = df["predicted_risk"].mean()
        return (
            f"The average predicted risk is {avg:,.0f}"
            if language == "English" else
            f"متوسط المخاطر المتوقعة هو {avg:,.0f}"
        )

    # Top 3 risky suppliers
    if any(word in q for word in ["top", "top 3", "أكثر", "3"]):
        top3 = df.nlargest(3, "predicted_risk")[["supplier_name", "predicted_risk"]]
        lines = [f"{row['supplier_name']}: {row['predicted_risk']:,.0f}" for _, row in top3.iterrows()]
        return (
            "Top 3 risky suppliers:\n" + "\n".join(lines)
            if language == "English" else
            "أعلى 3 موردين مخاطرة:\n" + "\n".join(lines)
        )

    # Threshold based queries
    if threshold is not None:
        # Risk threshold
        if any(word in q for word in risk_kw):
            filtered = df[df["predicted_risk"] > threshold]
            if filtered.empty:
                return (
                    "No suppliers found above that risk level."
                    if language == "English" else
                    "لا يوجد موردين فوق مستوى المخاطر هذا."
                )
            lines = [f"{row['supplier_name']}: {row['predicted_risk']:,.0f}" for _, row in filtered.iterrows()]
            return (
                f"Suppliers with risk above {threshold:,.0f}:\n" + "\n".join(lines)
                if language == "English" else
                f"الموردين الذين لديهم مخاطر أعلى من {threshold:,.0f}:\n" + "\n".join(lines)
            )
        # Delay threshold
        if any(word in q for word in delay_kw) and "delay_days" in df.columns:
            filtered = df[df["delay_days"] > threshold]
            if filtered.empty:
                return (
                    "No suppliers found above that delay days."
                    if language == "English" else
                    "لا يوجد موردين لديهم تأخير أعلى من ذلك."
                )
            lines = [f"{row['supplier_name']}: {int(row['delay_days'])} days" for _, row in filtered.iterrows()]
            return (
                f"Suppliers with delay days over {int(threshold)}:\n" + "\n".join(lines)
                if language == "English" else
                f"الموردين الذين لديهم تأخير أكثر من {int(threshold)} يوم:\n" + "\n".join(lines)
            )
        # FX exposure threshold
        if any(word in q for word in fx_kw) and "fx_exposure" in df.columns:
            filtered = df[df["fx_exposure"] > threshold]
            if filtered.empty:
                return (
                    "No suppliers with FX exposure above that value."
                    if language == "English" else
                    "لا يوجد موردين لديهم تعرض للعملات بهذا المستوى."
                )
            lines = [f"{row['supplier_name']}: {row['fx_exposure']:.0f}%" for _, row in filtered.iterrows()]
            return (
                f"Suppliers with FX exposure above {threshold:.0f}%:\n" + "\n".join(lines)
                if language == "English" else
                f"الموردين الذين لديهم تعرض للعملات أكثر من {threshold:.0f}%:\n" + "\n".join(lines)
            )
        # Commodity dependence threshold
        if any(word in q for word in commodity_kw) and "commodity_dependence" in df.columns:
            filtered = df[df["commodity_dependence"] > threshold]
            if filtered.empty:
                return (
                    "No suppliers with commodity dependence above that value."
                    if language == "English" else
                    "لا يوجد موردين لديهم اعتماد على السلع بهذا المستوى."
                )
            lines = [f"{row['supplier_name']}: {row['commodity_dependence']:.0f}%" for _, row in filtered.iterrows()]
            return (
                f"Suppliers with commodity dependence above {threshold:.0f}%:\n" + "\n".join(lines)
                if language == "English" else
                f"الموردين الذين لديهم اعتماد على السلع أكثر من {threshold:.0f}%:\n" + "\n".join(lines)
            )

    # Fallback for unrecognised queries
    if language == "English":
        if mode == "Beginner":
            return "I'm sorry, I couldn't understand your question. Try typing '?' to see some examples."
        else:
            return "Query not recognised. Try '?' for available shortcuts."
    else:
        # Arabic fallback
        if mode == "Beginner":
            return "عذرًا، لم أفهم سؤالك. اكتب '?' لرؤية بعض الأمثلة."
        else:
            return "لم يتم التعرف على الاستعلام. جرب '?' للاختصارات المتاحة."


def build_network_graph(df: pd.DataFrame, t: dict) -> go.Figure:
    """Build a Plotly network graph from supplier risk data.

    The graph links suppliers in descending order of risk to show the
    relative magnitude of their predicted risk.  Node sizes and colours
    scale with the average risk.  The translation dictionary ``t`` should
    contain ``risk`` and ``risk_level`` keys.

    Args:
        df: DataFrame with ``supplier_name`` and ``predicted_risk`` columns.
        t: Translation dictionary with ``risk`` and ``risk_level`` labels.
    Returns:
        A Plotly Figure representing the network.
    """
    G = nx.DiGraph()
    top_suppliers = df.groupby("supplier_name")["predicted_risk"].mean().sort_values(ascending=False).head(10)
    suppliers = list(top_suppliers.index)
    for s in suppliers:
        supplier_risk = df[df["supplier_name"] == s]["predicted_risk"].mean()
        G.add_node(s, risk=supplier_risk)
    for i in range(len(suppliers) - 1):
        G.add_edge(suppliers[i], suppliers[i + 1])
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="gray"), hoverinfo="none", mode="lines")
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    max_risk_val = max([G.nodes[n]["risk"] for n in G.nodes()]) if G.nodes() else 1
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        risk_val = G.nodes[node]["risk"]
        node_text.append(f"{node}<br>{t['risk']}: {risk_val:,.0f}")
        node_color.append(risk_val)
        node_size.append(20 + (risk_val / max_risk_val) * 40)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Reds",
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(text=t["risk_level"], side="right"),
                xanchor="left"
            )
        )
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _sanitize_pdf_text(text: str) -> str:
    """Sanitize strings for PDF output.

    FPDF uses latin‑1 encoding.  This helper replaces common Unicode
    punctuation (non‑breaking hyphens, en/em dashes, arrows, ellipsis) with
    ASCII equivalents and discards unsupported characters to avoid
    ``UnicodeEncodeError`` during PDF generation.

    Args:
        text: Arbitrary string.
    Returns:
        A sanitized string safe for FPDF.
    """
    if not isinstance(text, str):
        return text
    replacements = {
        "\u2011": "-",  # non‑breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2192": "->",  # rightwards arrow
        "\u2190": "<-",  # leftwards arrow
        "\u2191": "^",   # upwards arrow
        "\u2193": "v",   # downwards arrow
        "\u2026": "...",  # ellipsis
        "\u00A0": " ",   # non‑breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# -----------------------------------------------------------------------------
# Database helper functions
# -----------------------------------------------------------------------------

def init_db(db_path: str = "logifi.db") -> sqlite3.Connection:
    """Initialise the SQLite database and ensure required tables exist.

    Creates a lightweight database in the working directory. Two tables are
    created if they do not already exist:

    * ``users`` – storing account information (username, password, account_type,
      small_business flag, personal_mode flag).
    * ``suppliers`` – storing supplier details, aggregated ratings and risk
      metrics derived from uploaded CSVs.

    Args:
        db_path: File path for the SQLite database. Defaults to ``logifi.db``.
    Returns:
        A ``sqlite3.Connection`` object.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    # Users table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            account_type TEXT,
            small_business INTEGER DEFAULT 0,
            personal_mode INTEGER DEFAULT 0
        );
        """
    )
    # Suppliers table. Ratings and comments are aggregated in this table to
    # minimise query complexity for user accounts.
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS suppliers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            supplier_name TEXT UNIQUE,
            category TEXT,
            location TEXT,
            average_rating REAL DEFAULT 0,
            rating_count INTEGER DEFAULT 0,
            comments TEXT,
            low_moq INTEGER DEFAULT 0,
            predicted_risk REAL DEFAULT 0,
            delay_days REAL,
            fx_exposure REAL,
            commodity_dependence REAL
        );
        """
    )
    conn.commit()
    return conn


def add_suppliers_to_db(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert or update supplier records in the database based on a DataFrame.

    Each row in ``df`` represents a supplier. The function attempts to update
    existing records (by ``supplier_name``) with new risk metrics and other
    attributes; if a supplier does not already exist, a new record is
    inserted. Ratings and comments fields are preserved during updates.

    Args:
        df: DataFrame containing at minimum a ``supplier_name`` column. Optional
            columns include ``category``, ``location``, ``predicted_risk``,
            ``delay_days``, ``fx_exposure``, ``commodity_dependence``, and
            ``low_moq``.
        conn: SQLite connection.
    """
    if df is None or df.empty:
        return
    cur = conn.cursor()
    for _, row in df.iterrows():
        name = row.get("supplier_name") or f"Supplier"
        category = row.get("category", "General")
        location = row.get("location", "Unknown")
        predicted_risk = float(row.get("predicted_risk", 0) or 0)
        delay = row.get("delay_days")
        fx = row.get("fx_exposure")
        commodity = row.get("commodity_dependence")
        low_moq = int(row.get("low_moq", 0))
        # Upsert logic: update existing row preserving ratings, else insert new
        cur.execute("SELECT id, average_rating, rating_count, comments FROM suppliers WHERE supplier_name = ?", (name,))
        res = cur.fetchone()
        if res:
            supplier_id, avg_rating, rating_count, comments = res
            cur.execute(
                """
                UPDATE suppliers SET category=?, location=?, predicted_risk=?, delay_days=?, fx_exposure=?, commodity_dependence=?, low_moq=?
                WHERE id=?
                """,
                (category, location, predicted_risk, delay, fx, commodity, low_moq, supplier_id)
            )
        else:
            cur.execute(
                """
                INSERT INTO suppliers (supplier_name, category, location, predicted_risk, delay_days, fx_exposure, commodity_dependence, low_moq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, category, location, predicted_risk, delay, fx, commodity, low_moq)
            )
    conn.commit()


def rate_supplier_in_db(supplier_name: str, rating: int, comment: str, conn: sqlite3.Connection) -> None:
    """Update a supplier's aggregated rating and append a comment.

    Calculates a new weighted average rating based on existing average and count.
    Stores the aggregated comments as a concatenated string separated by
    delimiters. In a production setting, comments would be stored in a
    separate table to avoid string concatenation, but this approach
    simplifies the MVP.

    Args:
        supplier_name: Name of the supplier to rate.
        rating: Integer rating between 1 and 5.
        comment: Comment text provided by the user.
        conn: SQLite connection.
    """
    if rating < 1 or rating > 5:
        return
    cur = conn.cursor()
    cur.execute("SELECT id, average_rating, rating_count, comments FROM suppliers WHERE supplier_name = ?", (supplier_name,))
    res = cur.fetchone()
    if not res:
        return
    supplier_id, avg_rating, count, comments = res
    count = count or 0
    avg_rating = avg_rating or 0
    new_count = count + 1
    new_avg = ((avg_rating * count) + rating) / new_count
    # Append comment to comments field
    new_comments = (comments or "") + f"||{comment.strip()}" if comment else comments
    cur.execute(
        """
        UPDATE suppliers SET average_rating=?, rating_count=?, comments=? WHERE id=?
        """,
        (new_avg, new_count, new_comments, supplier_id)
    )
    conn.commit()


def fetch_suppliers(conn: sqlite3.Connection, category: Optional[str] = None, location: Optional[str] = None) -> pd.DataFrame:
    """Fetch supplier records from the database with optional filters.

    Args:
        conn: SQLite connection.
        category: Optional category filter. If provided, only suppliers in
            this category are returned.
        location: Optional location filter. If provided, only suppliers in
            this location are returned.
    Returns:
        A pandas DataFrame of suppliers.
    """
    cur = conn.cursor()
    query = "SELECT supplier_name, category, location, average_rating, rating_count, comments, low_moq, predicted_risk FROM suppliers"
    params = []
    conditions = []
    if category:
        conditions.append("category = ?")
        params.append(category)
    if location:
        conditions.append("location = ?")
        params.append(location)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    cur.execute(query, params)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["supplier_name", "category", "location", "average_rating", "rating_count", "comments", "low_moq", "predicted_risk"])
    return df


# -----------------------------------------------------------------------------
# Chat assistant helper functions
# -----------------------------------------------------------------------------

def get_help_suggestions(language: str, mode: str, t: dict) -> list:
    """Return a list of example questions when the user enters '?'.

    For beginners, suggestions focus on definitions and simple queries. For
    professionals, suggestions highlight numeric shortcuts and thresholds.

    Args:
        language: "English" or "العربية".
        mode: "Beginner" or "Pro".
        t: Current translation dictionary.
    Returns:
        A list of suggestion strings in the appropriate language.
    """
    suggestions = []
    if language == "English":
        if mode == "Beginner":
            suggestions = [
                t["suggestion_explain"].format(term="risk"),
                t["suggestion_explain"].format(term="FX exposure"),
                t["suggestion_explain"].format(term="commodity dependence"),
                t["suggestion_delay_threshold"].format(days=10),
                t["suggestion_risk_threshold"].format(threshold="50k"),
            ]
        else:
            suggestions = [
                t["suggestion_top"],
                t["suggestion_risk_threshold"].format(threshold="50k"),
                t["suggestion_fx_threshold"].format(fx=30),
                t["suggestion_commodity_threshold"].format(commodity=20),
                t["suggestion_delay_threshold"].format(days=10),
            ]
    else:
        # Arabic suggestions
        if mode == "Beginner":
            suggestions = [
                t["suggestion_explain"].format(term="المخاطر"),
                t["suggestion_explain"].format(term="التعرض للعملة"),
                t["suggestion_explain"].format(term="الاعتماد على السلع"),
                t["suggestion_delay_threshold"].format(days=10),
                t["suggestion_risk_threshold"].format(threshold="50k"),
            ]
        else:
            suggestions = [
                t["suggestion_top"],
                t["suggestion_risk_threshold"].format(threshold="50k"),
                t["suggestion_fx_threshold"].format(fx=30),
                t["suggestion_commodity_threshold"].format(commodity=20),
                t["suggestion_delay_threshold"].format(days=10),
            ]
    return suggestions


def generate_followup_suggestions(df: pd.DataFrame, language: str, mode: str, t: dict) -> list:
    """Generate follow‑up suggestions after answering a query.

    Suggestions encourage users to explore additional insights and prompt
    action‑oriented questions such as generating action plans or exploring
    thresholds. The suggestions are contextual: the first suggestion
    recommends creating an action plan for the highest‑risk supplier.

    Args:
        df: DataFrame containing at least ``predicted_risk`` and ``supplier_name``.
        language: "English" or "العربية".
        mode: "Beginner" or "Pro".
        t: Translation dictionary for constructing messages.
    Returns:
        A list of suggestion strings.
    """
    # Ensure predicted_risk exists
    if "predicted_risk" not in df.columns:
        try:
            df["predicted_risk"] = df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
        except Exception:
            df["predicted_risk"] = 0
    # Identify the supplier with the highest predicted risk
    try:
        idx = df["predicted_risk"].idxmax()
        top_supplier = df.loc[idx].get("supplier_name", f"Supplier {idx+1}")
    except Exception:
        top_supplier = ""
    # Build suggestions using translation templates
    suggestions = []
    # Action plan suggestion
    if top_supplier:
        suggestions.append(t["suggestion_action_plan"].format(supplier=top_supplier))
    # Threshold based suggestions
    # Choose threshold values heuristically; use integers as strings so that translation remains intact
    suggestions.append(t["suggestion_risk_threshold"].format(threshold="50k"))
    suggestions.append(t["suggestion_delay_threshold"].format(days=10))
    # Include FX and commodity if available
    if "fx_exposure" in df.columns:
        suggestions.append(t["suggestion_fx_threshold"].format(fx=30))
    if "commodity_dependence" in df.columns:
        suggestions.append(t["suggestion_commodity_threshold"].format(commodity=20))
    # Add top 3 suggestion for pro users or as final
    suggestions.append(t["suggestion_top"])
    # For beginners, also include a glossary suggestion
    if mode == "Beginner":
        # Suggest asking about risk definition
        if language == "English":
            suggestions.append(t["suggestion_explain"].format(term="risk"))
        else:
            suggestions.append(t["suggestion_explain"].format(term="المخاطر"))
    # Return a limited number of suggestions to avoid clutter
    return suggestions[:5]


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------

def main() -> None:
    # Configure Streamlit page and theme
    st.set_page_config(page_title="LogiFi – Where Logistics Meets Finance", layout="wide")
    # Apply custom CSS for brand colours
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0A192F;
            color: #F2F2F2;
        }
        .stSidebar {
            background-color: #0D263B;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #DAA520;
        }
        .css-1v0mbdj .stButton>button {
            background-color: #DAA520;
            color: #0A192F;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use a background image for the login page
    if not st.session_state.get("logged_in", False):
        bg_path = os.path.join(os.path.dirname(__file__), "login_bg.png")
        if os.path.exists(bg_path):
            with open(bg_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{b64}");
                    background-size: cover;
                    background-position: center;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    # Sidebar: logo and language selection
    try:
        st.sidebar.image("logo.png", use_container_width=True)
    except TypeError:
        st.sidebar.image("logo.png", use_column_width=True)
    language = st.sidebar.selectbox("🌐 Language", ["English", "العربية"])

    # Translation dictionary for UI elements
    translations = {
        "English": {
            "home": "Home",
            "upload": "Data Upload",
            "analysis": "Risk Analysis",
            "chat": "Chat Assistant",
            "logout": "Logout",
            "login": "Login",
            "username": "Username",
            "password": "Password",
            "login_btn": "Login",
            "create_account": "Create Account",
            "welcome": "Welcome to LogiFi",
            "before": "Before",
            "after": "After",
            "before_text": "Manual risk tracking -> delayed insight -> reactive decisions -> cash tied‑up",
            "after_text": "Input scenario -> instantaneous risk estimate -> visual risk map -> suggested actions -> download report -> speak aloud results",
            "upload_title": "Upload your supply‑chain data",
            "analysis_title": "Current Supply Chain Risk Overview",
            "chat_title": "Ask questions about your supply‑chain risk data",
            "no_data": "No data uploaded yet.",
            "send": "Send",
            "alerts_title": "Alerts & What‑if Problems",
            "problem_prompt": "Select a problem",
            "port_delay": "Port Delay",
            "currency_fluctuation": "Currency Fluctuation",
            "commodity_spike": "Commodity Price Spike",
            "extra_days": "Additional delay days",
            "extra_fx": "Additional FX exposure (%)",
            "extra_commodity": "Additional commodity dependence (%)",
            "impact_message": "If {problem} increases by {value}, you could lose approximately {impact:,.0f} SAR.",
            # Home page enhancements
            "hero_title": "LogiFi",
            "hero_tagline": "Transforming supply chain risk management",
            "feature1_title": "Real-Time Insights",
            "feature1_desc": "Instant risk estimations and visualizations.",
            "feature2_title": "Scenario Simulation",
            "feature2_desc": "Test hypothetical scenarios and foresee impacts.",
            "feature3_title": "Intelligent Recommendations",
            "feature3_desc": "Actionable guidance to mitigate risks."
            ,
            # Chat assistant additions
            "mode_label": "Chat Mode",
            "beginner": "Beginner",
            "pro": "Pro",
            "help_text": "💡 Don’t know what to ask? Type '?' to see possibilities.",
            "follow_up": "You might also ask:",
            "suggestion_action_plan": "Generate an action plan for {supplier}",
            "suggestion_risk_threshold": "Show suppliers with risk above {threshold}",
            "suggestion_delay_threshold": "Show suppliers with delay days > {days}",
            "suggestion_fx_threshold": "Show suppliers with FX exposure above {fx}%",
            "suggestion_commodity_threshold": "Show suppliers with commodity dependence above {commodity}%",
            "suggestion_top": "Show top 3 risky suppliers",
            "suggestion_explain": "What is {term}?"
            ,
            # New strings for dual-account system and user pages
            "account_type": "Choose Account Type",
            "business_account": "Business",
            "user_account": "User",
            "small_business": "Small Business Mode",
            "personal_mode": "Personal Use Mode",
            "login_or_signup": "Login or Sign Up",
            "signup_btn": "Sign Up",
            "account_exists": "Account already exists.",
            "create_success": "Account created successfully! Please log in.",
            "no_suppliers": "No suppliers available.",
            "search_supplier": "Search suppliers...",
            "filter_category": "Filter by category",
            "filter_location": "Filter by location",
            "product_query": "Enter a product or category",
            "find_suppliers": "Find Suppliers",
            "top_results": "Top Results",
            "supplier_ratings": "Supplier Ratings",
            "supply_chain_finder": "Supply Chain Finder",
            "education_title": "Glossary",
            "education_desc": "Expand the sections below to learn about supply chain terms.",
            "rate_suppliers": "Rate Suppliers"
            ,
            # New keys for the enhanced landing page and hero banner
            "slogan_simple": "Simplicity. Inclusivity. Contingency.",
            "slogan_subtitle": "Supply chain intelligence for everyone.",
            "bridging_tagline": "Bridging supply chain risk management for businesses and everyday users.",
            "key_features": "Key Features",
            "feature_risk_modelling": "AI‑powered supply chain risk modelling",
            "feature_ratings": "Supplier ratings and recommendations",
            "feature_finder": "Personalised supply chain finder for small businesses"
        },
        "العربية": {
            "home": "الرئيسية",
            "upload": "رفع البيانات",
            "analysis": "تحليل المخاطر",
            "chat": "المساعد النصي",
            "logout": "تسجيل الخروج",
            "login": "تسجيل الدخول",
            "username": "اسم المستخدم",
            "password": "كلمة المرور",
            "login_btn": "تسجيل الدخول",
            "create_account": "إنشاء حساب",
            "welcome": "مرحبًا بكم في LogiFi",
            "before": "قبل",
            "after": "بعد",
            "before_text": "تتبع المخاطر يدويًا -> تأخر في المعلومات -> قرارات تفاعلية -> تجميد السيولة",
            "after_text": "إدخال سيناريو -> تقدير فوري للمخاطر -> خريطة مخاطر مرئية -> إجراءات مقترحة -> تحميل التقرير -> قراءة النتائج صوتيًا",
            "upload_title": "قم برفع بيانات سلسلة التوريد الخاصة بك",
            "analysis_title": "نظرة عامة على مخاطر سلسلة التوريد الحالية",
            "chat_title": "اطرح أسئلة حول بيانات مخاطر سلسلة التوريد",
            "no_data": "لم يتم رفع أي بيانات حتى الآن.",
            "send": "إرسال",
            "alerts_title": "التنبيهات والمشاكل المحتملة",
            "problem_prompt": "اختر مشكلة",
            "port_delay": "تأخير الميناء",
            "currency_fluctuation": "تقلب العملة",
            "commodity_spike": "ارتفاع أسعار السلع",
            "extra_days": "أيام التأخير الإضافية",
            "extra_fx": "زيادة التعرض للعملة (%)",
            "extra_commodity": "زيادة الاعتماد على السلع (%)",
            "impact_message": "إذا زاد {problem} بمقدار {value}، قد تخسر حوالي {impact:,.0f} ريال.",
            # Home page enhancements
            "hero_title": "LogiFi",
            "hero_tagline": "تحويل إدارة مخاطر سلسلة التوريد",
            "feature1_title": "رؤى فورية",
            "feature1_desc": "تقديرات فورية للمخاطر ورؤى مرئية.",
            "feature2_title": "محاكاة السيناريو",
            "feature2_desc": "اختبر سيناريوهات افتراضية وتوقع التأثيرات.",
            "feature3_title": "توصيات ذكية",
            "feature3_desc": "إرشادات عملية لتقليل المخاطر."
            ,
            # Chat assistant additions
            "mode_label": "وضع الدردشة",
            "beginner": "مبتدئ",
            "pro": "محترف",
            "help_text": "💡 لا تعرف ماذا تسأل؟ اكتب '?' لرؤية الاحتمالات.",
            "follow_up": "يمكنك أيضًا أن تسأل:",
            "suggestion_action_plan": "إنشاء خطة عمل لـ {supplier}",
            "suggestion_risk_threshold": "عرض الموردين الذين لديهم مخاطر أعلى من {threshold}",
            "suggestion_delay_threshold": "عرض الموردين الذين لديهم تأخير أكبر من {days} يوم",
            "suggestion_fx_threshold": "عرض الموردين الذين لديهم تعرض للعملة أكثر من {fx}%",
            "suggestion_commodity_threshold": "عرض الموردين الذين لديهم اعتماد على السلع أكثر من {commodity}%",
            "suggestion_top": "عرض أعلى 3 موردين مخاطرة",
            "suggestion_explain": "ما هو {term}؟"
            ,
            # New strings for dual-account system and user pages
            "account_type": "اختر نوع الحساب",
            "business_account": "تجاري",
            "user_account": "مستخدم",
            "small_business": "وضع الأعمال الصغيرة",
            "personal_mode": "وضع الاستخدام الشخصي",
            "login_or_signup": "تسجيل الدخول أو إنشاء حساب",
            "signup_btn": "إنشاء حساب",
            "account_exists": "الحساب موجود بالفعل.",
            "create_success": "تم إنشاء الحساب بنجاح! الرجاء تسجيل الدخول.",
            "no_suppliers": "لا توجد موردين متاحين.",
            "search_supplier": "ابحث عن الموردين...",
            "filter_category": "تصفية حسب الفئة",
            "filter_location": "تصفية حسب الموقع",
            "product_query": "أدخل منتجًا أو فئة",
            "find_suppliers": "ابحث عن الموردين",
            "top_results": "أفضل النتائج",
            "supplier_ratings": "تقييمات الموردين",
            "supply_chain_finder": "الباحث عن سلسلة التوريد",
            "education_title": "قاموس المصطلحات",
            "education_desc": "قم بتوسيع الأقسام أدناه لتتعرف على مصطلحات سلسلة التوريد.",
            "rate_suppliers": "تقييم الموردين"
            ,
            # New keys for the enhanced landing page and hero banner
            "slogan_simple": "البساطة. الشمولية. الطوارئ.",
            "slogan_subtitle": "ذكاء سلسلة التوريد للجميع.",
            "bridging_tagline": "جسر إدارة مخاطر سلسلة التوريد بين الشركات والأفراد.",
            "key_features": "الميزات الرئيسية",
            "feature_risk_modelling": "نمذجة المخاطر القائمة على الذكاء الاصطناعي لسلسلة التوريد",
            "feature_ratings": "تقييمات الموردين والتوصيات",
            "feature_finder": "باحث مخصص لسلسلة التوريد للأعمال الصغيرة"
        }
    }
    t = translations[language]

    # ---------------------------------------------------------------------
    # Chat experience level selector (Beginner vs Pro)
    # Place this after translations are defined so that we can use the
    # translated labels from the dictionary. We store both the display mode
    # (in the user's language) and an internal representation (English) in
    # Streamlit's session state.
    # ---------------------------------------------------------------------
    if "chat_mode_display" not in st.session_state:
        # Default to beginner mode in the user's language
        st.session_state.chat_mode_display = t["beginner"]
    # Render the radio selector; using a fixed key ensures the state is
    # preserved between reruns
    st.session_state.chat_mode_display = st.sidebar.radio(
        label=t["mode_label"],
        options=[t["beginner"], t["pro"]],
        index=0 if st.session_state.chat_mode_display == t["beginner"] else 1,
        key="mode_selector"
    )
    # Derive the internal chat mode (English words) for logic based on the
    # selected display value. Store it so that it can be accessed in the
    # chat page logic below.
    if st.session_state.chat_mode_display == t["pro"]:
        st.session_state.internal_chat_mode = "Pro"
    else:
        st.session_state.internal_chat_mode = "Beginner"

    # -----------------------------------------------------------------
    # Initialise database connection and account/session variables
    # -----------------------------------------------------------------
    # Create or connect to the SQLite database (stored in working directory).
    if "conn" not in st.session_state:
        st.session_state.conn = init_db()
    conn = st.session_state.conn

    # Initialise top‑level session state variables if they don't exist.
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.account_type = ""  # 'business' or 'user'
        st.session_state.account_choice = None  # selected on landing page
        st.session_state.small_business = False
        st.session_state.personal_mode = False
        st.session_state.user_data = None  # data uploaded by business accounts
        st.session_state.chat_history = []  # chat memory for business chat

    # Helper to rerun the app when login state changes
    def _safe_rerun():
        # Streamlit's rerun functions may throw during script execution; catch errors gracefully
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
                return
            except Exception:
                pass
        if hasattr(st, "rerun"):
            try:
                st.rerun()
                return
            except Exception:
                pass
        return

    # -----------------------------------------------------------------
    # Account selection and authentication flows
    # -----------------------------------------------------------------
    if not st.session_state.logged_in:
        # Landing page: let the user choose an account type (business vs user)
        if st.session_state.account_choice is None:
            # Hero section: display the brand name and slogan as styled text
            st.markdown(
                f"""
                <div style="text-align:center; padding:60px 20px;">
                    <h1 style="font-size:4rem; color:#DAA520; margin-bottom:1rem; font-weight:bold;">LogiFi</h1>
                    <p style="font-size:2.5rem; color:#DAA520; margin:0; font-weight:bold; line-height:1.2;">
                        {t.get('slogan_simple', 'Simplicity. Inclusivity. Contingency.').replace('.', '<br>')}
                    </p>
                    <p style="font-size:1.2rem; color:#CCCCCC; margin-top:1rem;">
                        {t.get('slogan_subtitle', 'Supply chain intelligence for everyone.')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Bridging tagline beneath the hero banner
            st.markdown(
                f"""
                <p style="text-align:center; font-size:1.1rem; color:#BBBBBB; margin-top:20px;">
                    {t.get('bridging_tagline', 'Bridging supply chain risk management for businesses and everyday users.')}
                </p>
                """,
                unsafe_allow_html=True,
            )
            # Account selection heading
            st.markdown(
                f"""
                <h3 style="text-align:center; margin-top:30px;">{t.get('account_type', 'Choose Account Type')}</h3>
                """,
                unsafe_allow_html=True,
            )
            # Account selection buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t.get("business_account", "Business")):
                    st.session_state.account_choice = "business"
                    _safe_rerun()
            with col2:
                if st.button(t.get("user_account", "User")):
                    st.session_state.account_choice = "user"
                    _safe_rerun()
            # Key features summary below buttons
            st.markdown(
                f"""
                <div style="margin-top:40px;">
                    <h4 style="color:#DAA520;">{t.get('key_features', 'Key Features')}</h4>
                    <ul style="line-height:1.6; color:#DDDDDD;">
                        <li>{t.get('feature_risk_modelling', 'AI‑powered supply chain risk modelling')}</li>
                        <li>{t.get('feature_ratings', 'Supplier ratings and recommendations')}</li>
                        <li>{t.get('feature_finder', 'Personalised supply chain finder for small businesses')}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        else:
            # Render login / sign up form for the chosen account type
            acc_type = st.session_state.account_choice
            acc_label = t.get("business_account", "Business") if acc_type == "business" else t.get("user_account", "User")
            st.markdown(f"<h2>{acc_label} {t.get('login_or_signup', 'Login or Sign Up')}</h2>", unsafe_allow_html=True)
            mode_option = st.selectbox(t.get("login_or_signup", "Login or Sign Up"), [t.get("login", "Login"), t.get("signup_btn", "Sign Up")])
            username = st.text_input(t.get("username", "Username"), key="auth_user")
            password = st.text_input(t.get("password", "Password"), type="password", key="auth_pass")
            # Additional options for user accounts during sign up
            small_business = False
            personal_use = False
            if acc_type == "user" and mode_option == t.get("signup_btn", "Sign Up"):
                small_business = st.checkbox(t.get("small_business", "Small Business Mode"), value=False, key="sb_mode")
                personal_use = st.checkbox(t.get("personal_mode", "Personal Use Mode"), value=False, key="pu_mode")
            if st.button(mode_option):
                if not username or not password:
                    st.error(t.get("no_data", "Please fill in all fields."))
                else:
                    cur = conn.cursor()
                    if mode_option == t.get("signup_btn", "Sign Up"):
                        # Check if user exists
                        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
                        if cur.fetchone():
                            st.error(t.get("account_exists", "Account already exists."))
                        else:
                            # Insert new user
                            cur.execute(
                                "INSERT INTO users (username, password, account_type, small_business, personal_mode) VALUES (?,?,?,?,?)",
                                (username, password, acc_type, int(small_business), int(personal_use)),
                            )
                            conn.commit()
                            st.success(t.get("create_success", "Account created successfully! Please log in."))
                    else:
                        # Login path
                        cur.execute(
                            "SELECT account_type, small_business, personal_mode FROM users WHERE username = ? AND password = ?",
                            (username, password),
                        )
                        row = cur.fetchone()
                        if row:
                            account_type, sb, pu = row
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.account_type = account_type
                            st.session_state.small_business = bool(sb)
                            st.session_state.personal_mode = bool(pu)
                            # Reset user_data and chat history on login
                            st.session_state.user_data = None
                            st.session_state.chat_history = []
                            _safe_rerun()
                        else:
                            st.error("Invalid credentials")
            # Option to go back to account selection
            if st.button("⬅️ Back"):
                st.session_state.account_choice = None
                _safe_rerun()
            return

    # -----------------------------------------------------------------
    # Authenticated: render dashboards based on account type
    # -----------------------------------------------------------------
    acc_type = st.session_state.account_type
    # Business dashboard navigation
    if acc_type == "business":
        # Sidebar navigation for business users
        rate_label = t.get("rate_suppliers", "Rate Suppliers")
        nav_options = [t.get("upload", "Data Upload"), t.get("analysis", "Risk Analysis"), rate_label, t.get("chat", "Chat Assistant"), t.get("logout", "Logout")]
        page = st.sidebar.radio("", nav_options)
        # Upload data page
        if page == t.get("upload", "Data Upload"):
            st.markdown(f"<h2>{t.get('upload_title', 'Upload your supply‑chain data')}</h2>", unsafe_allow_html=True)
            uploaded = st.file_uploader("CSV", type=["csv"])
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    if "supplier_name" not in df.columns:
                        df["supplier_name"] = [f"Supplier {i+1}" for i in range(len(df))]
                    # Compute predicted risk
                    df["predicted_risk"] = df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
                    # Determine low_moq flag heuristically if 'moq' column exists
                    if "moq" in df.columns:
                        threshold = df["moq"].median()
                        df["low_moq"] = (df["moq"] <= threshold).astype(int)
                    else:
                        df["low_moq"] = 0
                    # Add to DB
                    add_suppliers_to_db(df, conn)
                    # Save to session for risk analysis
                    st.session_state.user_data = df
                    st.success("File uploaded and processed successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
            elif st.session_state.user_data is None:
                st.info(t.get("no_data", "No data uploaded yet."))
            else:
                st.dataframe(st.session_state.user_data.head())
        # Risk analysis & recommendations page
        elif page == t.get("analysis", "Risk Analysis"):
            st.markdown(f"<h2>{t.get('analysis_title', 'Current Supply Chain Risk Overview')}</h2>", unsafe_allow_html=True)
            df = st.session_state.user_data
            if df is None:
                st.info(t.get("no_data", "No data uploaded yet."))
            else:
                df = df.copy()
                # Ensure predicted risk column
                if "predicted_risk" not in df.columns:
                    df["predicted_risk"] = df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
                avg_risk = df["predicted_risk"].mean()
                highest_risk = df["predicted_risk"].max()
                st.metric("Avg Risk", f"{avg_risk:,.0f}")
                st.metric("Max Risk", f"{highest_risk:,.0f}")
                st.dataframe(df)
                # Scenario simulator
                with st.expander("🔍 Simulate a Scenario"):
                    st.write("Enter hypothetical parameters to estimate risk.")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        inv = st.number_input("Inventory Value", min_value=0.0, value=float(df["inventory_value"].mean()) if "inventory_value" in df.columns else 0.0, format="%.2f")
                        suppliers_count = st.number_input("Number of Suppliers", min_value=1, value=int(df["num_suppliers"].mean()) if "num_suppliers" in df.columns else 1)
                    with col2:
                        rev = st.number_input("Monthly Revenue", min_value=0.0, value=float(df["monthly_revenue"].mean()) if "monthly_revenue" in df.columns else 0.0, format="%.2f")
                        delay = st.number_input("Delay Days", min_value=0, value=int(df["delay_days"].mean()) if "delay_days" in df.columns else 0)
                    with col3:
                        fx = st.number_input("FX Exposure (%)", min_value=0, max_value=100, value=int(df["fx_exposure"].mean()) if "fx_exposure" in df.columns else 0)
                        commodity = st.number_input("Commodity Dependence (%)", min_value=0, max_value=100, value=int(df["commodity_dependence"].mean()) if "commodity_dependence" in df.columns else 0)
                    if st.button("Estimate Risk"):
                        scenario = dict(
                            inventory_value=inv,
                            monthly_revenue=rev,
                            num_suppliers=suppliers_count,
                            delay_days=delay,
                            fx_exposure=fx,
                            commodity_dependence=commodity,
                        )
                        risk_val = calculate_risk(scenario)
                        st.metric("Estimated Risk", f"{risk_val:,.0f}")
                # Alerts and what-if problems
                with st.expander(f"📣 {t.get('alerts_title', 'Alerts & What‑if Problems')}"):
                    problem = st.selectbox(
                        t.get("problem_prompt", "Select a problem"),
                        [t.get("port_delay", "Port Delay"), t.get("currency_fluctuation", "Currency Fluctuation"), t.get("commodity_spike", "Commodity Price Spike")]
                    )
                    extra = None
                    if problem == t.get("port_delay", "Port Delay"):
                        extra = st.number_input(t.get("extra_days", "Additional delay days"), min_value=1, value=5)
                    elif problem == t.get("currency_fluctuation", "Currency Fluctuation"):
                        extra = st.number_input(t.get("extra_fx", "Additional FX exposure (%)"), min_value=1, max_value=100, value=10)
                    elif problem == t.get("commodity_spike", "Commodity Price Spike"):
                        extra = st.number_input(t.get("extra_commodity", "Additional commodity dependence (%)"), min_value=1, max_value=100, value=10)
                    if st.button("Estimate Impact") and extra is not None:
                        baseline_cost = df["predicted_risk"].sum()
                        new_df = df.copy()
                        if problem == t.get("port_delay", "Port Delay"):
                            new_df["delay_days"] = new_df["delay_days"] + extra
                        elif problem == t.get("currency_fluctuation", "Currency Fluctuation"):
                            new_df["fx_exposure"] = (new_df["fx_exposure"] + extra).clip(0, 100)
                        elif problem == t.get("commodity_spike", "Commodity Price Spike"):
                            new_df["commodity_dependence"] = (new_df["commodity_dependence"] + extra).clip(0, 100)
                        new_df["predicted_risk"] = new_df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
                        new_cost = new_df["predicted_risk"].sum()
                        impact = max(new_cost - baseline_cost, 0)
                        st.warning(t.get("impact_message", "If {problem} increases by {value}, you could lose approximately {impact:,.0f} SAR.").format(problem=problem, value=extra, impact=impact))
                # Network graph
                st.markdown("### 🌐 Risk Network")
                fig = build_network_graph(df, {"risk": "Risk", "risk_level": "Risk Level"})
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except TypeError:
                    st.plotly_chart(fig)
                # Static recommendations
                st.markdown("### 🧠 Recommendations")
                st.write("- Diversify suppliers to reduce risk")
                st.write("- Hedge FX exposure during high volatility")
                st.write("- Monitor commodity prices closely")
                # PDF report generation as in original application
                st.markdown("### 📝 Select Insights for Report")
                include_highest = st.checkbox("Include highest-risk supplier", value=True)
                include_avg = st.checkbox("Include average risk", value=True)
                include_over_50k = st.checkbox("Include suppliers over 50k risk", value=False)
                include_delay = st.checkbox("Include suppliers with delay days > 10", value=False)
                class PDFReport(FPDF):
                    def header(self):
                        if os.path.exists("logo.png"):
                            self.image("logo.png", x=10, y=8, w=25)
                        self.set_font("Arial", "B", 14)
                        self.cell(0, 10, _sanitize_pdf_text("LogiFi Supply Chain Risk Report"), ln=False, align="C")
                        self.set_font("Arial", "", 10)
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        self.cell(-20, 10, date_str, ln=True, align="R")
                        self.ln(5)
                if st.button("Download PDF Report"):
                    pdf = PDFReport()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, _sanitize_pdf_text("Current Supply Chain Risk Overview"), ln=True)
                    pdf.set_font("Arial", "", 12)
                    summary_text = (
                        f"Average Predicted Risk: {avg_risk:,.0f} SAR\n"
                        f"Highest Predicted Risk: {highest_risk:,.0f} SAR\n\n"
                        "This report summarizes the financial and operational risks in your supply chain."
                    )
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(summary_text))
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, _sanitize_pdf_text("Top Risky Suppliers"), ln=True)
                    pdf.set_font("Arial", "", 12)
                    top_suppliers_df = df.nlargest(5, "predicted_risk")[["supplier_name", "predicted_risk"]]
                    for _, row in top_suppliers_df.iterrows():
                        line = f"{row['supplier_name']}: {row['predicted_risk']:,.0f} SAR"
                        pdf.cell(0, 10, _sanitize_pdf_text(line), ln=True)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, _sanitize_pdf_text("Insights"), ln=True)
                    pdf.set_font("Arial", "", 12)
                    if include_highest:
                        pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Highest-risk supplier: {top_suppliers_df.iloc[0]['supplier_name']}"))
                    if include_avg:
                        pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Average risk across suppliers: {avg_risk:,.0f} SAR"))
                    if include_over_50k:
                        over_50k_list = df[df["predicted_risk"] > 50_000]["supplier_name"].tolist()
                        suppliers_str = ", ".join(over_50k_list) if over_50k_list else "None"
                        pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Suppliers with risk > 50K SAR: {suppliers_str}"))
                    if include_delay:
                        delayed_list = df[df["delay_days"] > 10]["supplier_name"].tolist()
                        delayed_str = ", ".join(delayed_list) if delayed_list else "None"
                        pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Suppliers with delay > 10 days: {delayed_str}"))
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, _sanitize_pdf_text("AI-Powered Action Plan"), ln=True)
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(
                        "Short-Term: Diversify suppliers and hedge FX exposure\n"
                        "Medium-Term: Negotiate better terms with suppliers\n"
                        "Long-Term: Invest in supply-chain digitization"
                    ))
                    savings_df = df[["supplier_name", "predicted_risk"]].copy()
                    savings_df["expected_savings"] = savings_df["predicted_risk"] * 0.30
                    total_savings = savings_df["expected_savings"].sum()
                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, _sanitize_pdf_text("Expected Savings"), ln=True)
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(
                        f"If you implement the above actions, a 30% reduction in risk could save approximately {total_savings:,.0f} SAR across your supply chain."
                    ))
                    fig_bar, ax = plt.subplots(figsize=(6, 3))
                    top_suppliers_df.plot(kind="bar", x="supplier_name", y="predicted_risk", ax=ax, legend=False)
                    ax.set_ylabel("Risk (SAR)")
                    ax.set_title("Top 5 Risky Suppliers")
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format="png")
                    plt.close(fig_bar)
                    buf.seek(0)
                    chart_path = "temp_chart.png"
                    with open(chart_path, "wb") as f:
                        f.write(buf.read())
                    pdf.image(chart_path, x=10, y=None, w=180)
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    pdf_output = f"logifi_risk_report_{date_str}.pdf"
                    pdf.output(pdf_output)
                    with open(pdf_output, "rb") as f:
                        st.download_button("📄 Download PDF", f, file_name=pdf_output, mime="application/pdf")
        # Rate suppliers page
        elif page == rate_label:
            st.markdown(f"<h2>{t.get('rate_suppliers', 'Rate Suppliers')}</h2>", unsafe_allow_html=True)
            # Fetch suppliers from DB
            suppliers_df = fetch_suppliers(conn)
            if suppliers_df.empty:
                st.info(t.get("no_suppliers", "No suppliers available."))
            else:
                supplier_names = suppliers_df["supplier_name"].tolist()
                selected_name = st.selectbox("Select Supplier", supplier_names)
                rating = st.slider("Rating", min_value=1, max_value=5, value=5)
                comment = st.text_area("Comment", "", height=80)
                if st.button("Submit Rating"):
                    rate_supplier_in_db(selected_name, rating, comment, conn)
                    st.success("Thank you for your feedback!")
        # Chat assistant page (business only)
        elif page == t.get("chat", "Chat Assistant"):
            st.markdown(f"<h2>{t.get('chat_title', 'Ask questions about your supply‑chain risk data')}</h2>", unsafe_allow_html=True)
            df = st.session_state.user_data
            if df is None:
                st.info(t.get("no_data", "No data uploaded yet."))
            else:
                # Display previous conversation history
                for speaker, text_msg in st.session_state.chat_history:
                    st.markdown(f"**{speaker}:** {text_msg}")
                # Input box for new query
                user_query = st.text_input(
                    label="", value="", placeholder=t.get("chat_title", "Ask a question"), key="chat_input_biz"
                )
                # Help hint below the input
                st.markdown(f"<div style='color:#aaaaaa;font-size:0.85rem;'>{t.get('help_text','💡 Type ? for help')}</div>", unsafe_allow_html=True)
                if st.button(t.get("send", "Send")):
                    query = user_query.strip()
                    if query:
                        st.session_state.chat_history.append(("You", query))
                        current_mode = st.session_state.get("internal_chat_mode", "Beginner")
                        if query in ["?", "؟"]:
                            suggestions = get_help_suggestions(language, current_mode, translations[language])
                            suggestion_text = "\n".join([f"- {s}" for s in suggestions])
                            st.session_state.chat_history.append(("Bot", suggestion_text))
                        else:
                            reply = analyze_question(query, df, language, mode=current_mode)
                            st.session_state.chat_history.append(("Bot", reply))
                            followups = generate_followup_suggestions(df, language, current_mode, translations[language])
                            if followups:
                                followup_text = "\n".join([f"- {s}" for s in followups])
                                st.session_state.chat_history.append(("Bot", f"{t.get('follow_up','You might also ask:')}\n{followup_text}"))
        # Logout page
        elif page == t.get("logout", "Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.account_type = ""
            st.session_state.user_data = None
            st.session_state.chat_history = []
            st.session_state.account_choice = None
            _safe_rerun()
        return
    # User dashboard navigation
    else:
        # Sidebar navigation for user accounts
        supplier_ratings_label = t.get("supplier_ratings", "Supplier Ratings")
        sc_finder_label = t.get("supply_chain_finder", "Supply Chain Finder")
        edu_label = t.get("education_title", "Glossary")
        nav_options_u = [supplier_ratings_label, sc_finder_label, edu_label, t.get("logout", "Logout")]
        page_u = st.sidebar.radio("", nav_options_u)
        # Supplier ratings page
        if page_u == supplier_ratings_label:
            st.markdown(f"<h2>{t.get('supplier_ratings', 'Supplier Ratings')}</h2>", unsafe_allow_html=True)
            # Filters
            df_suppliers = fetch_suppliers(conn)
            if df_suppliers.empty:
                st.info(t.get("no_suppliers", "No suppliers available."))
            else:
                categories = sorted(set(df_suppliers["category"].dropna()))
                locations = sorted(set(df_suppliers["location"].dropna()))
                col1, col2 = st.columns(2)
                with col1:
                    cat_filter = st.selectbox(t.get("filter_category", "Filter by category"), ["All"] + categories)
                with col2:
                    loc_filter = st.selectbox(t.get("filter_location", "Filter by location"), ["All"] + locations)
                search_term = st.text_input(t.get("search_supplier", "Search suppliers..."), "")
                filtered_df = df_suppliers.copy()
                if cat_filter != "All":
                    filtered_df = filtered_df[filtered_df["category"] == cat_filter]
                if loc_filter != "All":
                    filtered_df = filtered_df[filtered_df["location"] == loc_filter]
                if search_term:
                    filtered_df = filtered_df[filtered_df["supplier_name"].str.contains(search_term, case=False, na=False)]
                # Display results
                st.dataframe(filtered_df[["supplier_name", "category", "location", "average_rating", "rating_count"]])
        # Supply chain finder page
        elif page_u == sc_finder_label:
            st.markdown(f"<h2>{t.get('supply_chain_finder', 'Supply Chain Finder')}</h2>", unsafe_allow_html=True)
            query = st.text_input(t.get("product_query", "Enter a product or category"), "")
            if st.button(t.get("find_suppliers", "Find Suppliers")):
                df_suppliers = fetch_suppliers(conn)
                if df_suppliers.empty:
                    st.info(t.get("no_suppliers", "No suppliers available."))
                else:
                    results = df_suppliers[df_suppliers["category"].str.contains(query, case=False, na=False) | df_suppliers["supplier_name"].str.contains(query, case=False, na=False)]
                    # Apply small business mode: filter for low MOQ
                    if st.session_state.small_business:
                        results = results[results["low_moq"] == 1]
                    # Sort by rating descending
                    results = results.sort_values(by="average_rating", ascending=False)
                    # Limit results for personal mode
                    limit = 3 if st.session_state.personal_mode else 5
                    st.markdown(f"### {t.get('top_results', 'Top Results')}")
                    for _, row in results.head(limit).iterrows():
                        st.write(f"**{row['supplier_name']}** – {row['category']} – {row['location']}")
                        st.write(f"Rating: {row['average_rating']:.2f} (from {int(row['rating_count'])} reviews)")
                        if row['predicted_risk']:
                            st.write(f"Predicted Risk: {row['predicted_risk']:,.0f}")
                        st.markdown("---")
        # Education / glossary page
        elif page_u == edu_label:
            st.markdown(f"<h2>{t.get('education_title', 'Glossary')}</h2>", unsafe_allow_html=True)
            st.write(t.get("education_desc", "Expand the sections below to learn about supply chain terms."))
            # Use the definitions from analyze_question helper for glossary
            # Predefine a set of terms based on language
            terms_to_show = []
            if language == "English":
                terms_to_show = ["risk", "predicted risk", "fx exposure", "commodity dependence", "inventory value", "delay days"]
            else:
                terms_to_show = ["المخاطر", "المخاطر المتوقعة", "التعرض للعملة", "الاعتماد على السلع", "قيمة المخزون", "أيام التأخير"]
            for term in terms_to_show:
                with st.expander(term):
                    # Use analyze_question to get definition by prepending 'what is'
                    query_term = f"what is {term}" if language == "English" else f"ما هو {term}"
                    definition = analyze_question(query_term, pd.DataFrame(), language, mode="Beginner")
                    st.write(definition)
        # Logout for user
        elif page_u == t.get("logout", "Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.account_type = ""
            st.session_state.account_choice = None
            st.session_state.user_data = None
            st.session_state.chat_history = []
            _safe_rerun()
        return

    # Navigation
    page = st.sidebar.radio("", [t["home"], t["upload"], t["analysis"], t["chat"], t["logout"]])

    # Logout
    if page == t["logout"]:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_data = None
        st.session_state.chat_history = []
        _safe_rerun()

    # Home page
    if page == t["home"]:
        # Hero and features
        st.markdown(
            f"""
            <style>
            .hero-container {{
                text-align: center;
                padding: 60px 20px;
            }}
            .hero-container h1 {{
                font-size: 3rem;
                margin-bottom: 0.5rem;
                color: #DAA520;
            }}
            .hero-container p {{
                font-size: 1.2rem;
                color: #cccccc;
                margin-bottom: 2rem;
            }}
            .features-row {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                gap: 1rem;
                margin-top: 40px;
            }}
            .feature-card {{
                background-color: rgba(255,255,255,0.05);
                border-radius: 8px;
                padding: 20px;
                flex: 1 1 250px;
                max-width: 300px;
                box-shadow: 0 0 10px rgba(0,0,0,0.3);
            }}
            .feature-card h3 {{
                margin-top: 0;
                color: #DAA520;
                font-size: 1.4rem;
            }}
            .feature-card p {{
                color: #bbbbbb;
                font-size: 0.9rem;
                line-height: 1.4;
            }}
            @media (max-width: 768px) {{
                .features-row {{
                    flex-direction: column;
                    align-items: center;
                }}
            }}
            </style>
            <div class="hero-container">
                <h1>{t['hero_title']}</h1>
                <p>{t['hero_tagline']}</p>
            </div>
            <div class="features-row">
                <div class="feature-card">
                    <h3>{t['feature1_title']}</h3>
                    <p>{t['feature1_desc']}</p>
                </div>
                <div class="feature-card">
                    <h3>{t['feature2_title']}</h3>
                    <p>{t['feature2_desc']}</p>
                </div>
                <div class="feature-card">
                    <h3>{t['feature3_title']}</h3>
                    <p>{t['feature3_desc']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Data upload page
    elif page == t["upload"]:
        st.markdown(f"<h2>{t['upload_title']}</h2>", unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if "supplier_name" not in df.columns:
                    df["supplier_name"] = [f"Supplier {i+1}" for i in range(len(df))]
                st.session_state.user_data = df
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        elif st.session_state.user_data is None:
            st.info(t["no_data"])
        else:
            st.dataframe(st.session_state.user_data.head())

    # Risk analysis page
    elif page == t["analysis"]:
        st.markdown(f"<h2>{t['analysis_title']}</h2>", unsafe_allow_html=True)
        df = st.session_state.user_data
        if df is None:
            st.info(t["no_data"])
        else:
            df = df.copy()
            # Compute predicted risk for each row
            df["predicted_risk"] = df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
            avg_risk = df["predicted_risk"].mean()
            highest_risk = df["predicted_risk"].max()
            st.metric("Avg Risk", f"{avg_risk:,.0f}")
            st.metric("Max Risk", f"{highest_risk:,.0f}")
            st.dataframe(df)

            # Scenario simulator
            with st.expander("🔍 Simulate a Scenario"):
                st.write("Enter hypothetical parameters to estimate risk.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    inv = st.number_input("Inventory Value", min_value=0.0, value=float(df["inventory_value"].mean()), format="%.2f")
                    suppliers = st.number_input("Number of Suppliers", min_value=1, value=int(df["num_suppliers"].mean()))
                with col2:
                    rev = st.number_input("Monthly Revenue", min_value=0.0, value=float(df["monthly_revenue"].mean()), format="%.2f")
                    delay = st.number_input("Delay Days", min_value=0, value=int(df["delay_days"].mean()))
                with col3:
                    fx = st.number_input("FX Exposure (%)", min_value=0, max_value=100, value=int(df["fx_exposure"].mean()))
                    commodity = st.number_input("Commodity Dependence (%)", min_value=0, max_value=100, value=int(df["commodity_dependence"].mean()))
                if st.button("Estimate Risk"):
                    scenario = dict(
                        inventory_value=inv,
                        monthly_revenue=rev,
                        num_suppliers=suppliers,
                        delay_days=delay,
                        fx_exposure=fx,
                        commodity_dependence=commodity,
                    )
                    risk_val = calculate_risk(scenario)
                    st.metric("Estimated Risk", f"{risk_val:,.0f}")

            # Alerts and what-if problems
            with st.expander(f"📣 {t['alerts_title']}"):
                problem = st.selectbox(
                    t["problem_prompt"],
                    [t["port_delay"], t["currency_fluctuation"], t["commodity_spike"]]
                )
                extra = None
                if problem == t["port_delay"]:
                    extra = st.number_input(t["extra_days"], min_value=1, value=5)
                elif problem == t["currency_fluctuation"]:
                    extra = st.number_input(t["extra_fx"], min_value=1, max_value=100, value=10)
                elif problem == t["commodity_spike"]:
                    extra = st.number_input(t["extra_commodity"], min_value=1, max_value=100, value=10)
                if st.button("Estimate Impact") and extra is not None:
                    baseline_cost = df["predicted_risk"].sum()
                    new_df = df.copy()
                    if problem == t["port_delay"]:
                        new_df["delay_days"] = new_df["delay_days"] + extra
                    elif problem == t["currency_fluctuation"]:
                        new_df["fx_exposure"] = (new_df["fx_exposure"] + extra).clip(0, 100)
                    elif problem == t["commodity_spike"]:
                        new_df["commodity_dependence"] = (new_df["commodity_dependence"] + extra).clip(0, 100)
                    new_df["predicted_risk"] = new_df.apply(lambda row: calculate_risk(row.to_dict()), axis=1)
                    new_cost = new_df["predicted_risk"].sum()
                    impact = max(new_cost - baseline_cost, 0)
                    st.warning(t["impact_message"].format(problem=problem, value=extra, impact=impact))

            # Network graph
            st.markdown("### 🌐 Risk Network")
            fig = build_network_graph(df, {"risk": "Risk", "risk_level": "Risk Level"})
            try:
                st.plotly_chart(fig, use_container_width=True)
            except TypeError:
                st.plotly_chart(fig)

            # Static recommendations
            st.markdown("### 🧠 Recommendations")
            st.write("- Diversify suppliers to reduce risk")
            st.write("- Hedge FX exposure during high volatility")
            st.write("- Monitor commodity prices closely")

            # PDF report options and generation
            st.markdown("### 📝 Select Insights for Report")
            include_highest = st.checkbox("Include highest-risk supplier", value=True)
            include_avg = st.checkbox("Include average risk", value=True)
            include_over_50k = st.checkbox("Include suppliers over 50k risk", value=False)
            include_delay = st.checkbox("Include suppliers with delay days > 10", value=False)

            class PDFReport(FPDF):
                def header(self):
                    if os.path.exists("logo.png"):
                        self.image("logo.png", x=10, y=8, w=25)
                    self.set_font("Arial", "B", 14)
                    # Title
                    self.cell(0, 10, _sanitize_pdf_text("LogiFi Supply Chain Risk Report"), ln=False, align="C")
                    self.set_font("Arial", "", 10)
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    self.cell(-20, 10, date_str, ln=True, align="R")
                    self.ln(5)

            if st.button("Download PDF Report"):
                pdf = PDFReport()
                pdf.add_page()
                # Summary section
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, _sanitize_pdf_text("Current Supply Chain Risk Overview"), ln=True)
                pdf.set_font("Arial", "", 12)
                summary_text = (
                    f"Average Predicted Risk: {avg_risk:,.0f} SAR\n"
                    f"Highest Predicted Risk: {highest_risk:,.0f} SAR\n\n"
                    "This report summarizes the financial and operational risks in your supply chain."
                )
                pdf.multi_cell(0, 10, _sanitize_pdf_text(summary_text))
                # Top suppliers listing
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, _sanitize_pdf_text("Top Risky Suppliers"), ln=True)
                pdf.set_font("Arial", "", 12)
                top_suppliers_df = df.nlargest(5, "predicted_risk")[["supplier_name", "predicted_risk"]]
                for _, row in top_suppliers_df.iterrows():
                    line = f"{row['supplier_name']}: {row['predicted_risk']:,.0f} SAR"
                    pdf.cell(0, 10, _sanitize_pdf_text(line), ln=True)
                # Insights page
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, _sanitize_pdf_text("Insights"), ln=True)
                pdf.set_font("Arial", "", 12)
                if include_highest:
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Highest-risk supplier: {top_suppliers_df.iloc[0]['supplier_name']}"))
                if include_avg:
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Average risk across suppliers: {avg_risk:,.0f} SAR"))
                if include_over_50k:
                    over_50k_list = df[df["predicted_risk"] > 50_000]["supplier_name"].tolist()
                    suppliers_str = ", ".join(over_50k_list) if over_50k_list else "None"
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Suppliers with risk > 50K SAR: {suppliers_str}"))
                if include_delay:
                    delayed_list = df[df["delay_days"] > 10]["supplier_name"].tolist()
                    delayed_str = ", ".join(delayed_list) if delayed_list else "None"
                    pdf.multi_cell(0, 10, _sanitize_pdf_text(f"- Suppliers with delay > 10 days: {delayed_str}"))
                # Recommendations and savings
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, _sanitize_pdf_text("AI-Powered Action Plan"), ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, _sanitize_pdf_text(
                    "Short-Term: Diversify suppliers and hedge FX exposure\n"
                    "Medium-Term: Negotiate better terms with suppliers\n"
                    "Long-Term: Invest in supply-chain digitization"
                ))
                # Savings calculation
                savings_df = df[["supplier_name", "predicted_risk"]].copy()
                savings_df["expected_savings"] = savings_df["predicted_risk"] * 0.30
                total_savings = savings_df["expected_savings"].sum()
                pdf.ln(5)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, _sanitize_pdf_text("Expected Savings"), ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, _sanitize_pdf_text(
                    f"If you implement the above actions, a 30% reduction in risk could save approximately {total_savings:,.0f} SAR across your supply chain."
                ))
                # Chart of top 5 risks
                fig_bar, ax = plt.subplots(figsize=(6, 3))
                top_suppliers_df.plot(kind="bar", x="supplier_name", y="predicted_risk", ax=ax, legend=False)
                ax.set_ylabel("Risk (SAR)")
                ax.set_title("Top 5 Risky Suppliers")
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png")
                plt.close(fig_bar)
                buf.seek(0)
                chart_path = "temp_chart.png"
                with open(chart_path, "wb") as f:
                    f.write(buf.read())
                pdf.image(chart_path, x=10, y=None, w=180)
                # Save and provide download
                date_str = datetime.now().strftime("%Y-%m-%d")
                pdf_output = f"logifi_risk_report_{date_str}.pdf"
                pdf.output(pdf_output)
                with open(pdf_output, "rb") as f:
                    st.download_button("📄 Download PDF", f, file_name=pdf_output, mime="application/pdf")

    # Chat assistant page
    elif page == t["chat"]:
        st.markdown(f"<h2>{t['chat_title']}</h2>", unsafe_allow_html=True)
        df = st.session_state.user_data
        if df is None:
            st.info(t["no_data"])
        else:
            # Display previous conversation history
            for speaker, text_msg in st.session_state.chat_history:
                # Use markdown for consistent styling
                st.markdown(f"**{speaker}:** {text_msg}")
            # Input box for new query
            user_query = st.text_input(
                label="", value="", placeholder=t["chat_title"], key="chat_input"
            )
            # Help hint below the input
            st.markdown(f"<div style='color:#aaaaaa;font-size:0.85rem;'>{t['help_text']}</div>", unsafe_allow_html=True)
            if st.button(t["send"]):
                query = user_query.strip()
                if query:
                    # Append the user's message
                    st.session_state.chat_history.append(("You", query))
                    # Detect '?' help command (both English '?' and Arabic '؟')
                    # Determine the current internal chat mode (Beginner/Pro)
                    current_mode = st.session_state.get("internal_chat_mode", "Beginner")
                    if query in ["?", "؟"]:
                        suggestions = get_help_suggestions(language, current_mode, translations[language])
                        # Format suggestions with bullet points
                        suggestion_text = "\n".join([f"- {s}" for s in suggestions])
                        st.session_state.chat_history.append(("Bot", suggestion_text))
                    else:
                        # Generate answer using the current mode
                        reply = analyze_question(query, df, language, mode=current_mode)
                        st.session_state.chat_history.append(("Bot", reply))
                        # Generate follow‑up suggestions
                        followups = generate_followup_suggestions(df, language, current_mode, translations[language])
                        if followups:
                            followup_text = "\n".join([f"- {s}" for s in followups])
                            st.session_state.chat_history.append(("Bot", f"{t['follow_up']}\n{followup_text}"))


if __name__ == "__main__":
    main()