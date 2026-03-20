from flask import Flask, request, jsonify, send_from_directory, g, session, redirect, url_for
import pdfplumber
import joblib
import os
import re
import io
import sqlite3
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

# Credentials for hospital user access. Override with environment variables for production.
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "password")

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(BASE_DIR,"model","leukemia_model.pkl")
model=joblib.load(model_path)

UPLOAD_FOLDER=os.path.join(BASE_DIR,"uploads")
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

DB_PATH = os.environ.get(
    "PREDICTION_DB",
    os.path.join(BASE_DIR, "data", "predictions.db"),
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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
    # Initialize the SQLite database schema without requiring a request context.
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                file_name TEXT,
                record_id INTEGER,
                wbc REAL,
                rbc REAL,
                hb REAL,
                platelets REAL,
                probability REAL,
                risk TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# Initialize the database schema on startup.
# Flask 3.x removed before_first_request, so we initialize explicitly here.
init_db()


def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            # For browser access, send to the login page; for API, return JSON.
            if request.accept_mimetypes.accept_html:
                return redirect(url_for("login"))
            return jsonify({"error": "Authentication required"}), 401
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/login", methods=["GET", "POST"])
def login():
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
    # Note: frontend contains `loginpage.html`.
    if request.method == "GET":
        return send_from_directory(frontend_dir, "loginpage.html")

    # Expect JSON credentials for API / front-end usage.
    data = {}
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form or request.values

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        session.clear()
        session["logged_in"] = True
        session["user"] = username
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
    return jsonify({"user": session.get("user")})


@app.teardown_appcontext
def teardown_db(exception):
    close_db()


def record_prediction(source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk):
    db = get_db()
    db.execute(
        "INSERT INTO predictions (timestamp, source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk) "
        "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk),
    )
    db.commit()


def extract_cbc_from_text(text):
    def find(pattern):
        m=re.search(pattern,text)
        return float(m.group(1)) if m else 0.0

    wbc=find(r"WBC.*?([\d.]+)")
    rbc=find(r"RBC.*?([\d.]+)")
    hb=find(r"Hemoglobin.*?([\d.]+)")
    platelets=find(r"Platelet.*?([\d.]+)")

    return [wbc,rbc,hb,platelets]

def extract_cbc_from_pdf(path):
    text=""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t=page.extract_text()
            if t:
                text+=t+"\n"
    return extract_cbc_from_text(text)

def get_prediction(wbc, rbc, hb, platelets):
    """Make prediction with proper feature names to avoid sklearn warnings"""
    df = pd.DataFrame({
        "WBC": [wbc],
        "RBC": [rbc],
        "Hb": [hb],
        "Platelets": [platelets]
    })
    return float(model.predict_proba(df)[0][1])

def recommendation(prob):
    if prob<0.30:
        return "Low Risk – CBC values appear normal."
    elif prob<0.60:
        return "Medium Risk – Monitor and repeat CBC."
    else:
        return "High Risk – Possible leukemia pattern detected."


@app.route("/history", methods=["GET"])
@login_required
def get_history():
    """Return stored predictions (most recent first)."""
    db = get_db()
    rows = db.execute(
        "SELECT id, timestamp, source, file_name, record_id, wbc, rbc, hb, platelets, probability, risk "
        "FROM predictions ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()

    return jsonify([dict(row) for row in rows])

@app.route("/")
@login_required
def home():
    frontend_dir=os.path.abspath(os.path.join(BASE_DIR,"..","frontend"))
    return send_from_directory(frontend_dir,"index.html")

@app.route("/homepage.html")
@app.route("/homepage")
@login_required
def homepage():
    frontend_dir=os.path.abspath(os.path.join(BASE_DIR,"..","frontend"))
    return send_from_directory(frontend_dir,"homepage.html")

@app.route("/index.html")
@app.route("/index")
@login_required
def index_page():
    frontend_dir=os.path.abspath(os.path.join(BASE_DIR,"..","frontend"))
    return send_from_directory(frontend_dir,"index.html")

@app.route("/predict",methods=["POST"])
@login_required
def predict():
    results=[]
    total=0

    if request.is_json:
        data=request.get_json()

        w=float(data.get("WBC"))
        r=float(data.get("RBC"))
        h=float(data.get("Hb") or data.get("Hemoglobin"))
        p=float(data.get("Platelets"))

        prob=get_prediction(w, r, h, p)
        risk=recommendation(prob)

        record_prediction(
            source="manual",
            file_name=None,
            record_id=0,
            wbc=w,
            rbc=r,
            hb=h,
            platelets=p,
            probability=prob,
            risk=risk,
        )

        return jsonify({
            "total_records_processed": 1,
            "results": [
                {
                    "record_id": 0,
                    "probability": round(prob, 3),
                    "risk": risk,
                    "recommendation": risk
                }
            ]
        })

    if request.files:

        files=request.files.getlist("files") or list(request.files.values())

        for idx,f in enumerate(files):

            fname=f.filename
            lower=fname.lower()

            if lower.endswith(".pdf"):

                path=os.path.join(UPLOAD_FOLDER,fname)
                f.save(path)

                feats=extract_cbc_from_pdf(path)

                prob=get_prediction(feats[0], feats[1], feats[2], feats[3])
                risk=recommendation(prob)

                record_prediction(
                    source="file",
                    file_name=fname,
                    record_id=total,
                    wbc=feats[0],
                    rbc=feats[1],
                    hb=feats[2],
                    platelets=feats[3],
                    probability=prob,
                    risk=risk,
                )

                record_prediction(
                    source="file",
                    file_name=fname,
                    record_id=total,
                    wbc=feats[0],
                    rbc=feats[1],
                    hb=feats[2],
                    platelets=feats[3],
                    probability=prob,
                    risk=risk,
                )

                results.append({
                    "record_id": total,
                    "file": fname,
                    "probability": round(prob, 3),
                    "risk": risk,
                    "recommendation": risk
                })

                total += 1

            elif lower.endswith(".csv"):

                stream=io.StringIO(f.stream.read().decode("utf-8"))
                df=pd.read_csv(stream)

                for _,row in df.iterrows():

                    feats=[
                        float(row["WBC"]),
                        float(row["RBC"]),
                        float(row["Hb"]),
                        float(row["Platelets"])
                    ]

                    prob=get_prediction(feats[0], feats[1], feats[2], feats[3])
                    risk=recommendation(prob)

                    results.append({
                        "record_id": total,
                        "file": fname,
                        "probability": round(prob, 3),
                        "risk": risk,
                        "recommendation": risk
                    })

                    total += 1

        return jsonify({
            "total_records_processed":total,
            "results":results
        })

    return jsonify({"error":"No input provided"}),400

# Fallback route for any frontend path to prevent 404 errors.
@app.route('/<path:path>')
@login_required
def frontend_files(path):
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
    safe_path = os.path.normpath(path).lstrip(os.sep)
    candidate = os.path.join(frontend_dir, safe_path)
    # Prevent path traversal attacks.
    if os.path.commonpath([frontend_dir, candidate]) != frontend_dir:
        return send_from_directory(frontend_dir, "index.html")

    if os.path.exists(candidate) and os.path.isfile(candidate):
        return send_from_directory(frontend_dir, safe_path)

    # Fallback to the main dashboard (index) for unknown paths.
    return send_from_directory(frontend_dir, "index.html")

if __name__=="__main__":
    # Allow cloud platforms to control host/port via environment variables.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug)
    