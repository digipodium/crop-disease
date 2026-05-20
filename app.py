import os
import sqlite3
import base64
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (Flask, flash, g, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from plant_disease_model import PlantDiseasePredictor, cfg

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Initialize Flask app
app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY="change-this-secret-key",
    DATABASE=os.path.join(app.instance_path, "users.db"),
    UPLOAD_FOLDER=os.path.join(app.root_path, "static", "uploads"),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5 MB upload limit
)

# Ensure required folders exist
os.makedirs(app.instance_path, exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TEST_DIR = os.path.join(app.root_path, "test")

# Load the trained predictor once
try:
    predictor = PlantDiseasePredictor(cfg.MODEL_SAVE_PATH)
except Exception as e:
    predictor = None
    print(f"[Warning] Failed to load model predictor: {e}")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(
            app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

app.teardown_appcontext(close_db)


def init_db():
    db = get_db()
    user_table = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user'"
    ).fetchone()

    if user_table is None:
        db.execute(
            "CREATE TABLE user ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL, "
            "email TEXT UNIQUE NOT NULL, "
            "password TEXT NOT NULL"
            ")"
        )
        db.commit()
    else:
        columns = [row[1] for row in db.execute("PRAGMA table_info(user)").fetchall()]
        if "email" not in columns or "name" not in columns:
            db.execute("ALTER TABLE user RENAME TO user_old")
            db.execute(
                "CREATE TABLE user ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL, "
                "email TEXT UNIQUE NOT NULL, "
                "password TEXT NOT NULL"
                ")"
            )
            old_users = db.execute(
                "SELECT id, username, password FROM user_old"
            ).fetchall()
            for user in old_users:
                db.execute(
                    "INSERT INTO user (id, name, email, password) VALUES (?, ?, ?, ?)",
                    (user[0], user[1], user[1], user[2]),
                )
            db.execute("DROP TABLE user_old")
            db.commit()

    predictions_table = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
    ).fetchone()

    if predictions_table is None:
        db.execute(
            "CREATE TABLE predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER NOT NULL, "
            "image_filename TEXT NOT NULL, "
            "image_source TEXT NOT NULL, "
            "crop_type TEXT NOT NULL, "
            "disease_name TEXT NOT NULL, "
            "confidence REAL NOT NULL, "
            "timestamp TIMESTAMP NOT NULL, "
            "FOREIGN KEY(user_id) REFERENCES user(id)"
            ")"
        )
        db.commit()


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            "SELECT id, name, email FROM user WHERE id = ?", (user_id,)
        ).fetchone()


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_prediction(user_id, image_filename, image_source, disease_name, confidence):
    crop_type = disease_name.split()[0].title() if disease_name else "Unknown"
    timestamp = datetime.now()
    db = get_db()
    db.execute(
        "INSERT INTO predictions (user_id, image_filename, image_source, crop_type, disease_name, confidence, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, image_filename, image_source, crop_type, disease_name, confidence, timestamp),
    )
    db.commit()


def get_user_predictions(user_id):
    db = get_db()
    return db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,),
    ).fetchall()


def delete_user_prediction(user_id, prediction_id):
    db = get_db()
    db.execute(
        "DELETE FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, user_id),
    )
    db.commit()


def get_demo_images():
    samples = [
        {"filename": "black_rot.jpg", "label": "Apple Black Rot", "source": "test"},
        {"filename": "corn_healthy.jpg", "label": "Corn Healthy", "source": "uploads"},
        {"filename": "corn_northern_leaf_blight.jpg", "label": "Corn Northern Leaf Blight", "source": "test"},
        {"filename": "corn_northern_leaf.jpg", "label": "Corn Grey Leaf", "source": "test"},
        {"filename": "tomato-septoria-leaf-spot.jpg", "label": "Tomato Septoria Leaf Spot", "source": "test"},
        {"filename": "grape_black_rot.jpg", "label": "Grape Black Rot", "source": "test"},
        {"filename": "grape_esca_black_measles.jpg", "label": "Grape Esca Black Measles", "source": "uploads"},
        {"filename": "grape_healthy.jpg", "label": "Grape Healthy", "source": "uploads"},
        {"filename": "Grape_leaf_blight.jpg", "label": "Grape Leaf Blight", "source": "uploads"},
        {"filename": "tomato_late_blight.jpg", "label": "Tomato Late Blight", "source": "uploads"},
    ]

    demo_images = []
    uploads_dir = os.path.join(app.static_folder, "uploads")
    for sample in samples:
        if sample["source"] == "test":
            sample_path = os.path.join(TEST_DIR, sample["filename"])
        else:
            sample_path = os.path.join(uploads_dir, sample["filename"])

        if os.path.exists(sample_path):
            demo_images.append(sample)
    return demo_images


@app.route("/demo/image/<source>/<path:filename>")
def demo_image(source, filename):
    if source == "test":
        root = TEST_DIR
    elif source == "uploads":
        root = os.path.join(app.static_folder, "uploads")
    else:
        return redirect(url_for("upload"))

    safe_filename = os.path.basename(filename)
    return send_from_directory(root, safe_filename)


@app.route("/demo/predict/<source>/<path:filename>")
@login_required
def demo_predict(source, filename):
    if predictor is None:
        flash("Model is not loaded. Please start the app after the checkpoint is available.", "danger")
        return redirect(url_for("upload"))

    if source == "test":
        root = TEST_DIR
    elif source == "uploads":
        root = os.path.join(app.static_folder, "uploads")
    else:
        flash("Invalid demo image source.", "danger")
        return redirect(url_for("upload"))

    safe_filename = os.path.basename(filename)
    demo_path = os.path.join(root, safe_filename)
    if not os.path.exists(demo_path):
        flash("Demo image not found.", "danger")
        return redirect(url_for("upload"))

    try:
        result = predictor.predict(demo_path, top_k=5)
        result["predicted_class"] = result["predicted_class"].replace("___", " - ").replace("_", " ")
        for item in result["top_k_predictions"]:
            item["class"] = item["class"].replace("___", " - ").replace("_", " ")

        with open(demo_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        save_prediction(
            g.user["id"],
            safe_filename,
            source,
            result["predicted_class"],
            float(result["confidence"]),
        )

        return render_template(
            "results.html",
            result=result["predicted_class"],
            confidence=result["confidence"],
            image_data=image_data,
        )
    except Exception as e:
        flash(f"Demo prediction failed: {e}", "danger")
        return redirect(url_for("upload"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/history")
@login_required
def history():
    predictions = get_user_predictions(g.user["id"])
    return render_template("history.html", predictions=predictions)

@app.route("/history/delete/<int:prediction_id>")
@login_required
def delete_history(prediction_id):
    delete_user_prediction(g.user["id"], prediction_id)
    flash("Prediction record archived successfully.", "success")
    return redirect(url_for("history"))

@app.route("/upload", methods=("GET", "POST"))
@login_required
def upload():
    demo_images = get_demo_images()

    def render_upload():
        return render_template("upload.html", demo_images=demo_images)

    if predictor is None:
        flash("Model is not loaded. Please start the app after the checkpoint is available.", "danger")
        return render_upload()

    if request.method == "POST":
        if "leaf_image" not in request.files:
            flash("No image uploaded.", "danger")
            return render_upload()

        file = request.files["leaf_image"]
        if file.filename == "":
            flash("Please choose an image file.", "danger")
            return render_upload()

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            try:
                result = predictor.predict(save_path, top_k=5)
                result["predicted_class"] = result["predicted_class"].replace("___", " - ").replace("_", " ")
                for item in result["top_k_predictions"]:
                    item["class"] = item["class"].replace("___", " - ").replace("_", " ")

                with open(save_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")

                save_prediction(
                    g.user["id"],
                    filename,
                    "uploads",
                    result["predicted_class"],
                    float(result["confidence"]),
                )

                return render_template(
                    "results.html",
                    result=result["predicted_class"],
                    confidence=result["confidence"],
                    image_data=image_data,
                )
            except Exception as e:
                flash(f"Prediction failed: {e}", "danger")
                return render_upload()
        else:
            flash("Allowed image formats: jpg, jpeg, png.", "danger")
            return render_upload()

    return render_upload()


@app.route("/register", methods=("GET", "POST"))
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password", "")
        db = get_db()
        error = None

        if not name:
            error = "Full name is required."
        elif not email:
            error = "Email address is required."
        elif not password:
            error = "Password is required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif db.execute("SELECT id FROM user WHERE email = ?", (email,)).fetchone() is not None:
            error = "Email is already registered."

        if error is None:
            db.execute(
                "INSERT INTO user (name, email, password) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            db.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))

        flash(error, "danger")

    return render_template("register.html")


@app.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        db = get_db()
        error = None
        user = db.execute(
            "SELECT * FROM user WHERE email = ?", (email,)
        ).fetchone()

        if user is None:
            error = "Incorrect email or password."
        elif not check_password_hash(user["password"], password):
            error = "Incorrect email or password."

        if error is None:
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("upload"))

        flash(error, "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


@app.route("/predict", methods=("GET", "POST"))
@login_required
def predict():
    if predictor is None:
        flash("Model is not loaded. Please start the app after the checkpoint is available.", "danger")
        return render_template("upload.html", demo_images=get_demo_images())

    if request.method == "POST":
        if "leaf_image" not in request.files:
            flash("No image uploaded.", "danger")
            return redirect(request.url)

        file = request.files["leaf_image"]
        if file.filename == "":
            flash("Please choose an image file.", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            try:
                result = predictor.predict(save_path, top_k=5)
                result["predicted_class"] = result["predicted_class"].replace("___", " - ").replace("_", " ")
                for item in result["top_k_predictions"]:
                    item["class"] = item["class"].replace("___", " - ").replace("_", " ")
                save_prediction(
                    g.user["id"],
                    filename,
                    "uploads",
                    result["predicted_class"],
                    float(result["confidence"]),
                )
                return render_template("results.html", result=result["predicted_class"], confidence=result["confidence"], image_data=base64.b64encode(open(save_path, "rb").read()).decode("utf-8"))
            except Exception as e:
                flash(f"Prediction failed: {e}", "danger")
                return redirect(request.url)
        else:
            flash("Allowed image formats: jpg, jpeg, png.", "danger")
            return redirect(request.url)

    return render_template("upload.html", demo_images=get_demo_images())


class CurrentUser:
    def __init__(self, user):
        self._user = user

    @property
    def is_authenticated(self):
        return self._user is not None

    @property
    def name(self):
        return self._user["name"] if self._user else ""

    @property
    def id(self):
        return self._user["id"] if self._user else None


@app.context_processor
def inject_user():
    return {
        "user": g.user,
        "current_user": CurrentUser(g.user),
    }


if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
