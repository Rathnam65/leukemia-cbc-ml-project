import math
import os
import re
import sqlite3
from functools import wraps

import bcrypt
import joblib
import pandas as pd
import pdfplumber
from flask import Flask, g, jsonify, redirect, request, send_from_directory, session, url_for
from flask_cors import CORS


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
CORS(app, supports_credentials=True)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
)

AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "password")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
MODEL_PATH = os.path.join(BASE_DIR, "model", "leukemia_model.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.environ.get("PREDICTION_DB", os.path.join(BASE_DIR, "data", "predictions.db"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

try:
    print(f"Loading model from: {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        MODEL_FEATURE_COLUMNS = model_bundle.get(
            "feature_columns",
            ["WBC", "RBC", "Hb", "Platelets"],
        )
        MODEL_CLASS_LABELS = model_bundle.get("class_labels", [0, 1, 2])
    else:
        model = model_bundle
        MODEL_FEATURE_COLUMNS = ["WBC", "RBC", "Hb", "Platelets"]
        MODEL_CLASS_LABELS = [0, 1, 2]
    print("Model loaded successfully")
except Exception as exc:
    print(f"Model load error: {exc}")
    model = None
    MODEL_FEATURE_COLUMNS = ["WBC", "RBC", "Hb", "Platelets"]
    MODEL_CLASS_LABELS = [0, 1, 2]


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hospital TEXT,
                timestamp TEXT,
                source TEXT,
                file_name TEXT,
                record_id TEXT,
                wbc REAL,
                rbc REAL,
                hb REAL,
                platelets REAL,
                probability REAL,
                risk TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hospital_name TEXT,
                username TEXT UNIQUE,
                password BLOB
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"error": "Authentication required"}), 401
        return view(*args, **kwargs)

    return wrapped_view


def validate_input(wbc, rbc, hb, platelets):
    if not (1000 <= wbc <= 200000):
        return "Invalid WBC value"
    if not (1 <= rbc <= 10):
        return "Invalid RBC value"
    if not (3 <= hb <= 20):
        return "Invalid Hemoglobin value"
    if not (10000 <= platelets <= 1000000):
        return "Invalid Platelet count"
    return None


def risk_from_probability(probability):
    if probability <= 0.3:
        return "LOW RISK"
    if probability <= 0.6:
        return "MEDIUM RISK"
    return "HIGH RISK"


def label_to_risk(label):
    return {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK",
    }.get(int(label), "MEDIUM RISK")


def confidence_from_probability(probability):
    if probability > 0.85:
        return "High Confidence"
    if probability > 0.6:
        return "Moderate Confidence"
    return "Low Confidence"


def recommendation_for_risk(risk):
    if risk == "LOW RISK":
        return "Values appear within normal range."
    if risk == "MEDIUM RISK":
        return "Monitor patient and repeat CBC."
    return "High-risk indicators detected. Immediate evaluation recommended."


def get_reason(wbc, rbc, hb, platelets):
    reasons = []
    if wbc > 11000:
        reasons.append("Elevated WBC count")
    if wbc < 4000:
        reasons.append("Low WBC count")
    if hb < 12:
        reasons.append("Low hemoglobin")
    if platelets < 150000:
        reasons.append("Low platelet count")
    if platelets > 450000:
        reasons.append("High platelet count")
    return " | ".join(reasons) if reasons else "All parameters are within normal range"


def record_prediction(source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk):
    db = get_db()
    hospital = session.get("hospital", "default")
    db.execute(
        """
        INSERT INTO predictions (
            hospital, timestamp, source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk
        ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (hospital, source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk),
    )
    db.commit()


def extract_cbc_from_text(text):
    def find(pattern, group_index=1):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return 0.0
        value = match.group(group_index).replace(",", "").strip()
        return float(value)

    wbc = find(r"\bWBC\b.*?([\d,]+)")
    rbc = find(r"\bRBC\b.*?([\d.]+)")
    hb = find(r"(Hemoglobin|Hb).*?([\d.]+)", 2)
    platelets = find(r"Platelet[s]?\b.*?([\d,]+)")
    return [wbc, rbc, hb, platelets]


def extract_cbc_from_pdf(file_storage):
    text = ""
    with pdfplumber.open(file_storage) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return extract_cbc_from_text(text)


def engineer_model_features(wbc, rbc, hb, platelets):
    return {
        "WBC": wbc,
        "RBC": rbc,
        "Hb": hb,
        "Platelets": platelets,
        "WBC_log": math.log1p(max(wbc, 0.0)),
        "Platelets_log": math.log1p(max(platelets, 0.0)),
        "Hb_RBC_ratio": hb / max(rbc, 0.1),
        "WBC_Platelets_ratio": wbc / max(platelets, 1.0),
        "Cytopenia_count": int(rbc < 4.0) + int(hb < 12.0) + int(platelets < 150000),
        "Leukocytosis_score": int(wbc > 11000) + int(wbc > 20000) + int(wbc > 50000),
        "Anemia_score": int(hb < 12.0) + int(hb < 10.0) + int(hb < 8.0),
        "Thrombocytopenia_score": int(platelets < 150000) + int(platelets < 100000) + int(platelets < 50000),
    }


def clinical_risk_profile(wbc, rbc, hb, platelets):
    score = 0
    reasons = []

    if wbc > 50000:
        score += 4
        reasons.append("Severely elevated WBC count")
    elif wbc > 20000:
        score += 3
        reasons.append("Markedly elevated WBC count")
    elif wbc > 11000:
        score += 1.5
        reasons.append("Elevated WBC count")
    elif wbc < 4000:
        score += 1
        reasons.append("Low WBC count")

    if hb < 8:
        score += 3
        reasons.append("Severe anemia")
    elif hb < 10:
        score += 2
        reasons.append("Moderate anemia")
    elif hb < 12:
        score += 1
        reasons.append("Low hemoglobin")

    if rbc < 3:
        score += 2
        reasons.append("Severely low RBC count")
    elif rbc < 4:
        score += 1
        reasons.append("Low RBC count")

    if platelets < 50000:
        score += 3
        reasons.append("Severe thrombocytopenia")
    elif platelets < 100000:
        score += 2
        reasons.append("Markedly low platelet count")
    elif platelets < 150000:
        score += 1
        reasons.append("Low platelet count")
    elif platelets > 450000:
        score += 1
        reasons.append("High platelet count")

    cytopenias = int(rbc < 4.0) + int(hb < 12.0) + int(platelets < 150000)
    if cytopenias >= 3:
        score += 2
    elif cytopenias == 2:
        score += 1

    if score >= 6:
        risk = "HIGH RISK"
        probability = 0.88
        class_probs = {"LOW RISK": 0.02, "MEDIUM RISK": 0.10, "HIGH RISK": 0.88}
    elif score >= 3:
        risk = "MEDIUM RISK"
        probability = 0.65
        class_probs = {"LOW RISK": 0.15, "MEDIUM RISK": 0.65, "HIGH RISK": 0.20}
    else:
        risk = "LOW RISK"
        probability = 0.84
        class_probs = {"LOW RISK": 0.84, "MEDIUM RISK": 0.13, "HIGH RISK": 0.03}

    return {
        "risk": risk,
        "probability": probability,
        "class_probabilities": class_probs,
        "reasons": reasons,
    }


def has_severe_high_risk_evidence(wbc, rbc, hb, platelets):
    severe_markers = 0

    if wbc >= 50000:
        severe_markers += 1
    if hb <= 8:
        severe_markers += 1
    if rbc <= 3:
        severe_markers += 1
    if platelets <= 50000:
        severe_markers += 1

    # A very high WBC plus another meaningful abnormality is enough to allow HIGH risk.
    if wbc >= 30000 and (hb < 10 or rbc < 3.5 or platelets < 100000):
        return True

    return severe_markers >= 2


def get_prediction(wbc, rbc, hb, platelets):
    if model is None:
        raise RuntimeError("Model not loaded")

    feature_map = engineer_model_features(wbc, rbc, hb, platelets)
    df = pd.DataFrame([{column: feature_map[column] for column in MODEL_FEATURE_COLUMNS}])

    pred = int(model.predict(df)[0])
    raw_probs = model.predict_proba(df)[0]
    class_probabilities = {}
    for idx, label in enumerate(MODEL_CLASS_LABELS):
        class_probabilities[label_to_risk(label)] = float(raw_probs[idx])

    probability = float(class_probabilities.get(label_to_risk(pred), max(raw_probs)))
    return pred, probability, class_probabilities


def build_prediction_response(wbc, rbc, hb, platelets):
    heuristic = clinical_risk_profile(wbc, rbc, hb, platelets)
    risk = heuristic["risk"]
    probability = heuristic["probability"]
    class_probabilities = heuristic["class_probabilities"]

    if model is not None:
        try:
            pred, _model_probability, model_class_probs = get_prediction(wbc, rbc, hb, platelets)
            ordered_risks = ("LOW RISK", "MEDIUM RISK", "HIGH RISK")
            blended = {}
            for risk_name in ordered_risks:
                blended[risk_name] = round(
                    0.45 * heuristic["class_probabilities"].get(risk_name, 0.0)
                    + 0.55 * model_class_probs.get(risk_name, 0.0),
                    6,
                )

            risk = max(blended, key=blended.get)
            probability = float(blended[risk])
            class_probabilities = blended

            # Guardrail: keep severe multi-parameter abnormalities from being downgraded too far.
            if heuristic["risk"] == "HIGH RISK" and label_to_risk(pred) == "LOW RISK":
                risk = "HIGH RISK"
                probability = max(probability, heuristic["probability"])
                class_probabilities["HIGH RISK"] = max(
                    class_probabilities.get("HIGH RISK", 0.0),
                    heuristic["class_probabilities"]["HIGH RISK"],
                )
        except Exception:
            pass

    if risk == "HIGH RISK" and not has_severe_high_risk_evidence(wbc, rbc, hb, platelets):
        risk = "MEDIUM RISK"
        probability = max(
            class_probabilities.get("MEDIUM RISK", 0.0),
            heuristic["class_probabilities"].get("MEDIUM RISK", 0.0),
            0.60,
        )
        class_probabilities["HIGH RISK"] = min(class_probabilities.get("HIGH RISK", 0.0), 0.49)
        class_probabilities["MEDIUM RISK"] = max(class_probabilities.get("MEDIUM RISK", 0.0), probability)

    if risk == "LOW RISK" and heuristic["risk"] == "MEDIUM RISK":
        risk = "MEDIUM RISK"
        probability = max(probability, heuristic["probability"], 0.55)
        class_probabilities["MEDIUM RISK"] = max(class_probabilities.get("MEDIUM RISK", 0.0), probability)

    return {
        "probability": round(probability, 3),
        "risk": risk,
        "confidence": confidence_from_probability(probability),
        "reason": get_reason(wbc, rbc, hb, platelets),
        "recommendation": recommendation_for_risk(risk),
        "class_probabilities": {key: round(value, 3) for key, value in class_probabilities.items()},
    }


def normalize_password(stored_password):
    if isinstance(stored_password, memoryview):
        return stored_password.tobytes()
    if isinstance(stored_password, str):
        return stored_password.encode("utf-8")
    return stored_password


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return send_from_directory(FRONTEND_DIR, "signup.html")

    data = request.get_json(silent=True) or request.form
    hospital = (data.get("hospital_name") or "").strip()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not hospital or not username or not password:
        return jsonify({"error": "Missing fields"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    db = get_db()

    try:
        db.execute(
            "INSERT INTO users (hospital_name, username, password) VALUES (?, ?, ?)",
            (hospital, username, hashed),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "User already exists"}), 400

    return jsonify({"message": "Signup successful"})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return send_from_directory(FRONTEND_DIR, "loginpage.html")

    data = request.get_json(silent=True) if request.is_json else None
    data = data or request.form or request.values

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if user:
        stored_password = normalize_password(user["password"])
        if stored_password and bcrypt.checkpw(password.encode("utf-8"), stored_password):
            session.clear()
            session["logged_in"] = True
            session["user"] = username
            session["hospital"] = user["hospital_name"]
            return jsonify({"success": True})

    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        session.clear()
        session["logged_in"] = True
        session["user"] = username
        session["hospital"] = "default"
        return jsonify({"success": True})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    if request.accept_mimetypes.accept_html:
        return redirect(url_for("login"))
    return jsonify({"success": True})


@app.route("/whoami", methods=["GET"])
@login_required
def whoami():
    return jsonify({"user": session.get("user"), "hospital": session.get("hospital")})


@app.route("/history")
@login_required
def history():
    db = get_db()
    hospital = session.get("hospital")
    rows = db.execute(
        """
        SELECT * FROM predictions
        WHERE hospital = ?
        ORDER BY timestamp DESC
        LIMIT 100
        """,
        (hospital,),
    ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/predict", methods=["POST"])
@login_required
def predict_manual():
    data = request.get_json(silent=True) or {}

    try:
        wbc = float(data.get("wbc", 0))
        rbc = float(data.get("rbc", 0))
        hb = float(data.get("hb", 0))
        platelets = float(data.get("platelets", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input format"}), 400

    validation_error = validate_input(wbc, rbc, hb, platelets)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    result = build_prediction_response(wbc, rbc, hb, platelets)
    record_prediction(
        source="manual",
        file_name=None,
        record_id=None,
        wbc=wbc,
        rbc=rbc,
        hb=hb,
        platelets=platelets,
        probability=result["probability"],
        risk=result["risk"],
    )
    return jsonify(result)


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []

    for file in files:
        filename = (file.filename or "").lower()
        if not filename:
            continue

        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            normalized_columns = {col: col.strip().lower() for col in df.columns}
            df = df.rename(columns=normalized_columns)
            required_columns = {"wbc", "rbc", "hb", "platelets"}
            if not required_columns.issubset(df.columns):
                return jsonify({"error": f"CSV file {filename} is missing required CBC columns"}), 400
        elif filename.endswith(".pdf"):
            wbc, rbc, hb, platelets = extract_cbc_from_pdf(file)
            df = pd.DataFrame([{"wbc": wbc, "rbc": rbc, "hb": hb, "platelets": platelets}])
        else:
            return jsonify({"error": f"Unsupported file type: {filename}"}), 400

        for index, row in df.iterrows():
            try:
                wbc = float(row["wbc"])
                rbc = float(row["rbc"])
                hb = float(row["hb"])
                platelets = float(row["platelets"])
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid CBC values found in {filename}"}), 400

            validation_error = validate_input(wbc, rbc, hb, platelets)
            if validation_error:
                return jsonify({"error": f"{validation_error} in {filename}"}), 400

            result = build_prediction_response(wbc, rbc, hb, platelets)
            record_prediction(
                source="file",
                file_name=filename,
                record_id=str(index + 1),
                wbc=wbc,
                rbc=rbc,
                hb=hb,
                platelets=platelets,
                probability=result["probability"],
                risk=result["risk"],
            )
            results.append(
                {
                    "record_id": str(index + 1),
                    "probability": result["probability"],
                    "risk": result["risk"],
                    "confidence": result["confidence"],
                    "reason": result["reason"],
                    "recommendation": result["recommendation"],
                }
            )

    return jsonify({"total_records_processed": len(results), "results": results})


@app.teardown_appcontext
def teardown_db(exception):
    close_db(exception)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 10000))
    app.run(host=host, port=port)
