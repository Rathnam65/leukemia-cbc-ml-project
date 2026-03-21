from flask import Flask, request, jsonify, send_from_directory, g, session, redirect, url_for
import pdfplumber
import joblib
import os
import re
import io
import sqlite3
import pandas as pd
import bcrypt   # ✅ ADDED
from flask_cors import CORS


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
CORS(app, supports_credentials=True)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True
)
# Default admin login (kept)
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "password")

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(BASE_DIR,"model","leukemia_model.pkl")
try:
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
    print("✅ MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("❌ MODEL LOAD ERROR:", e)
    model = None

UPLOAD_FOLDER=os.path.join(BASE_DIR,"uploads")
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

DB_PATH = os.environ.get(
    "PREDICTION_DB",
    os.path.join(BASE_DIR, "data", "predictions.db"),
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ---------------- DB ----------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        # EXISTING TABLE
        conn.execute("""
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
""")

        # ✅ NEW USERS TABLE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hospital_name TEXT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)

        conn.commit()
    finally:
        conn.close()


init_db()


# ---------------- AUTH ----------------
def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"error": "Authentication required"}), 401
        return view(*args, **kwargs)

    return wrapped_view


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

    if request.method == "GET":
        return send_from_directory(frontend_dir, "signup.html")

    data = request.get_json() or request.form

    hospital = data.get("hospital_name")
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    db = get_db()

    try:
        db.execute(
            "INSERT INTO users (hospital_name, username, password) VALUES (?, ?, ?)",
            (hospital, username, hashed)
        )
        db.commit()
        return jsonify({"message": "Signup successful"})
    except:
        return jsonify({"error": "User already exists"}), 400


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

    if request.method == "GET":
        return send_from_directory(frontend_dir, "loginpage.html")

    data = request.get_json() if request.is_json else request.form or request.values

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    db = get_db()

    # ✅ CHECK DB USERS
    user = db.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()

    if user and bcrypt.checkpw(password.encode(), user["password"]):
        session.clear()
        session["logged_in"] = True
        session["user"] = username
        session["hospital"] = user["hospital_name"]
        return jsonify({"success": True})

    # ✅ FALLBACK ADMIN LOGIN
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
    return jsonify({
        "user": session.get("user"),
        "hospital": session.get("hospital")
    })

@app.route("/history")
@login_required
def history():
    db = get_db()
    hospital = session.get("hospital")

    rows = db.execute("""
        SELECT * FROM predictions
        WHERE hospital=?
        ORDER BY timestamp DESC
        LIMIT 100
    """, (hospital,)).fetchall()

    return jsonify([dict(r) for r in rows])

@app.teardown_appcontext
def teardown_db(exception):
    close_db()


# ---------------- ML + DB ----------------
def record_prediction(source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk):
    db = get_db()
    hospital = session.get("hospital", "default")

    db.execute(
        "INSERT INTO predictions (hospital, timestamp, source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk) "
        "VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (hospital, source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk),
    )
    db.commit()


def extract_cbc_from_text(text):
    def find(pattern, group_index=1):
        m = re.search(pattern, text)
        if m:
            value = m.group(group_index)
            value = value.replace(",", "")  # ✅ remove commas
            return float(value)
        return 0.0

    wbc = find(r"WBC.*?([\d,]+)")
    rbc = find(r"RBC.*?([\d.]+)")
    hb = find(r"(Hemoglobin|Hb).*?([\d.]+)", 2)
    platelets = find(r"Platelet.*?([\d,]+)")

    return [wbc, rbc, hb, platelets]


def extract_cbc_from_pdf(path):
    text=""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t=page.extract_text()
            if t:
                text+=t+"\n"
    return extract_cbc_from_text(text)


def get_prediction(wbc, rbc, hb, platelets):
    if model is None:
        raise Exception("Model not loaded")

    df = pd.DataFrame({
        "WBC": [wbc],
        "RBC": [rbc],
        "Hb": [hb],
        "Platelets": [platelets]
    })

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    return pred, max(probs)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    frontend_dir=os.path.abspath(os.path.join(BASE_DIR,"..","frontend"))
    return send_from_directory(frontend_dir,"index.html")
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
@app.route("/predict", methods=["POST"])
@login_required
def predict_manual():
    data = request.get_json()

    try:
        wbc = float(data.get("wbc", 0))
        rbc = float(data.get("rbc", 0))
        hb = float(data.get("hb", 0))
        platelets = float(data.get("platelets", 0))
    except:
        return jsonify({"error": "Invalid input format"}), 400

    # ✅ validation
    validation_error = validate_input(wbc, rbc, hb, platelets)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    # 🛡️ SAFETY OVERRIDE (VERY IMPORTANT)
    def is_normal(wbc, rbc, hb, platelets):
        return (
            4000 <= wbc <= 11000 and
            4.0 <= rbc <= 6.0 and
            12 <= hb <= 17 and
            150000 <= platelets <= 450000
        )

    if is_normal(wbc, rbc, hb, platelets):
        risk = "Low Risk"
        prob = 0.99
    else:
        pred, prob = get_prediction(wbc, rbc, hb, platelets)

        risk_map = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

        risk = risk_map[pred]

    # ✅ save to DB
    record_prediction(
        source="manual",
        file_name=None,
        record_id=None,
        wbc=wbc,
        rbc=rbc,
        hb=hb,
        platelets=platelets,
        probability=prob,
        risk=risk
    )

    # ✅ confidence label
    if prob > 0.85:
        confidence = "High Confidence"
    elif prob > 0.6:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence"

    return jsonify({
        "probability": round(prob, 3),
        "risk": risk,
        "confidence": confidence,
        "reason": get_reason(wbc, rbc, hb, platelets),
        "recommendation": (
            "Values appear within normal range." if risk == "LOW RISK"
            else "Monitor patient and repeat CBC." if risk == "MEDIUM RISK"
            else "High-risk indicators detected. Immediate evaluation recommended."
        )
    })
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []

    for file in files:
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            df.columns = [c.strip().capitalize() for c in df.columns]

        elif filename.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"

            wbc, rbc, hb, platelets = extract_cbc_from_text(text)

            df = pd.DataFrame([{
                "WBC": wbc,
                "RBC": rbc,
                "Hb": hb,
                "Platelets": platelets
            }])

        else:
            continue

        for _, row in df.iterrows():
            pred, prob = get_prediction(
        row["WBC"],
        row["RBC"],
        row["Hb"],
        row["Platelets"]
    )

            risk_map = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

            risk = risk_map[pred]

    # ✅ correct DB save
            record_prediction(
        source="file",
        file_name=filename,
        record_id=None,
        wbc=row["WBC"],
        rbc=row["RBC"],
        hb=row["Hb"],
        platelets=row["Platelets"],
        probability=prob,
        risk=risk
    )
            results.append({
                "probability": prob,
                "risk": risk,
                "recommendation": (
                   "Values appear within normal range." if risk == "LOW RISK"
                    else "Monitor patient and repeat CBC." if risk == "MEDIUM RISK"
                    else "Abnormal parameters detected. Further medical evaluation recommended."
                )
            })

    return jsonify({
        "total_records_processed": len(results),
        "results": results
    })
# ---------------- START ----------------
if __name__=="__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 10000))
    app.run(host=host, port=port)