"""
train_model.py — FruitScan (standalone, no features.py needed)
"""
import os, pickle, numpy as np, colorsys
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

IMG_SIZE              = (64, 64)
MAX_SAMPLES_PER_CLASS = 1500
N_ESTIMATORS          = 500
OUTPUT_DIR            = "model"

def extract_features(img_pil):
    """Histogram RGB+HSV (32 bin) + statistik + piksel 16x16. Total: 975 fitur."""
    img = img_pil.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img) / 255.0
    hists = []
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch], bins=32, range=(0,1))
        hists.append(h.astype(float) / (h.sum() + 1e-6))
    flat     = arr.reshape(-1, 3)
    hsv_flat = np.array([colorsys.rgb_to_hsv(r,g,b) for r,g,b in flat])
    for ch in range(3):
        h, _ = np.histogram(hsv_flat[:,ch], bins=32, range=(0,1))
        hists.append(h.astype(float) / (h.sum() + 1e-6))
    stats = []
    for ch in range(3):
        d = arr[:,:,ch].flatten()
        stats += [d.mean(), d.std(), float(np.median(d)),
                  float(np.percentile(d,25)), float(np.percentile(d,75))]
    small = np.array(img_pil.resize((16,16)).convert("RGB")) / 255.0
    return np.concatenate([np.concatenate(hists), np.array(stats), small.flatten()])

def load_dataset(split="train"):
    split_dir    = os.path.join(".", split)
    all_X, all_y = [], []
    classes = sorted([d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))])
    print(f"\n[INFO] Memuat '{split}' — kelas: {classes}")
    for cls in classes:
        folder = os.path.join(split_dir, cls)
        count  = 0
        for fname in sorted(os.listdir(folder)):
            if count >= MAX_SAMPLES_PER_CLASS: break
            try:
                img = Image.open(os.path.join(folder,fname)).convert("RGB")
                all_X.append(extract_features(img))
                all_y.append(cls); count += 1
            except: continue
        print(f"  ✓  {cls:20s} → {count} sampel")
    return np.array(all_X), all_y

def train():
    print("="*55)
    print("  🍎 FruitScan — Training Model")
    print(f"  Pohon: {N_ESTIMATORS} | Sampel/kelas: {MAX_SAMPLES_PER_CLASS}")
    print(f"  Fitur: 975 (Histogram RGB+HSV + Statistik + Piksel)")
    print("="*55)

    X_train, y_train_raw = load_dataset("train")
    X_test,  y_test_raw  = load_dataset("test")

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n[INFO] Latih: {len(X_train)} | Uji: {len(X_test)} | Fitur: {X_train.shape[1]}")
    print("[INFO] Melatih Random Forest...")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=None,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*55}")
    print(f"  ✅ AKURASI: {acc*100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/rf_model.pkl",      "wb") as f: pickle.dump(clf,    f)
    with open(f"{OUTPUT_DIR}/label_encoder.pkl", "wb") as f: pickle.dump(le,     f)
    with open(f"{OUTPUT_DIR}/scaler.pkl",        "wb") as f: pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/config.json",       "w")  as f:
        json.dump({"img_size":[64,64], "n_classes":len(le.classes_),
                   "class_names":list(le.classes_), "accuracy":round(acc,4),
                   "use_scaler":True}, f, indent=2)
    print("\n✅ Selesai! Jalankan: python app.py")

if __name__ == "__main__":
    train()