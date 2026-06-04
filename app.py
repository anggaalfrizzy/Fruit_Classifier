"""
app.py — FruitScan Flask Backend (standalone, no features.py needed)
"""

import os, json, pickle, uuid, numpy as np, colorsys
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "model")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = "fruit-rf-secret-2024"
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    paths = {
        "model":   os.path.join(MODEL_DIR, "rf_model.pkl"),
        "encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
        "config":  os.path.join(MODEL_DIR, "config.json"),
        "scaler":  os.path.join(MODEL_DIR, "scaler.pkl"),
    }
    if not all(os.path.exists(paths[k]) for k in ["model","encoder","config"]):
        return None, None, None, None
    with open(paths["model"],   "rb") as f: clf    = pickle.load(f)
    with open(paths["encoder"], "rb") as f: le     = pickle.load(f)
    with open(paths["config"],  "r")  as f: cfg    = json.load(f)
    scaler = None
    if os.path.exists(paths["scaler"]):
        with open(paths["scaler"], "rb") as f: scaler = pickle.load(f)
    return clf, le, cfg, scaler

clf, le, config, scaler = load_model()
MODEL_READY = clf is not None

# ── Feature extraction ────────────────────────────────────────────────────────
IMG_SIZE = (64, 64)

def extract_features(img_pil):
    """Histogram RGB+HSV (32 bin) + statistik + piksel 16x16. Total: 975 fitur."""
    img = img_pil.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img) / 255.0

    # Histogram RGB — 32 bin × 3 = 96
    hists = []
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch], bins=32, range=(0,1))
        h = h.astype(float) / (h.sum() + 1e-6)
        hists.append(h)

    # Histogram HSV — 32 bin × 3 = 96
    flat     = arr.reshape(-1, 3)
    hsv_flat = np.array([colorsys.rgb_to_hsv(r,g,b) for r,g,b in flat])
    for ch in range(3):
        h, _ = np.histogram(hsv_flat[:,ch], bins=32, range=(0,1))
        h = h.astype(float) / (h.sum() + 1e-6)
        hists.append(h)

    # Statistik per channel RGB: mean, std, median, p25, p75 → 15
    stats = []
    for ch in range(3):
        d = arr[:,:,ch].flatten()
        stats += [d.mean(), d.std(), float(np.median(d)),
                  float(np.percentile(d,25)), float(np.percentile(d,75))]

    # Piksel kecil 16×16 RGB → 768
    small = np.array(img_pil.resize((16,16)).convert("RGB")) / 255.0

    return np.concatenate([np.concatenate(hists), np.array(stats), small.flatten()])

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

def predict(filepath):
    if not MODEL_READY: return None
    img      = Image.open(filepath).convert("RGB")
    features = extract_features(img).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    pred_idx   = clf.predict(features)[0]
    pred_proba = clf.predict_proba(features)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(pred_proba[pred_idx]) * 100
    top5_idx   = np.argsort(pred_proba)[::-1][:5]
    top5 = [{"label": le.inverse_transform([i])[0],
              "confidence": round(float(pred_proba[i])*100, 2)} for i in top5_idx]
    return {"label": pred_label, "confidence": round(confidence,2), "top5": top5}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL_READY, config=config)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        flash("Tidak ada file yang dikirim.", "danger"); return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("Pilih file gambar terlebih dahulu.", "warning"); return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Format tidak didukung. Gunakan PNG, JPG, atau JPEG.", "danger"); return redirect(url_for("index"))
    ext      = secure_filename(file.filename).rsplit(".",1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    result = predict(filepath)
    if result is None:
        flash("Model belum tersedia. Jalankan train_model.py terlebih dahulu.", "danger")
        return redirect(url_for("index"))
    return render_template("result.html", result=result,
                           image_path=f"uploads/{filename}", config=config)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename): return jsonify({"error": "Invalid type"}), 400
    ext      = secure_filename(file.filename).rsplit(".",1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    result = predict(filepath)
    return jsonify(result) if result else jsonify({"error": "Model not loaded"}), 503

@app.route("/about")
def about():
    return render_template("about.html", config=config)

if __name__ == "__main__":
    print("="*50)
    print("  🍎 FruitScan — Random Forest + Flask")
    if MODEL_READY:
        print(f"  ✅ Model loaded | {config['n_classes']} kelas")
        print(f"  📊 Akurasi: {config['accuracy']*100:.2f}%")
    else:
        print("  ⚠️  Jalankan: python train_model.py")
    print("  🌐 http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True, host="0.0.0.0", port=5000)