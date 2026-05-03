from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import pandas as pd


app = Flask(__name__)

# ── Heart Disease Model ──────────────────────────────────────────────────────
HEART_MODEL_PATH = "diseaseheart_modelv1.pkl"
heart_model = None

if os.path.exists(HEART_MODEL_PATH):
    heart_model = joblib.load(HEART_MODEL_PATH)
    print("✅ Heart model loaded successfully.")
else:
    print("⚠️  Heart model file not found. Using mock predictions.")

# ── Brain Tumor Model ────────────────────────────────────────────────────────
brain_model = None

try:
    import tensorflow as tf
    BRAIN_MODEL_PATH_H5 = "brain_tumor_mobilenet.h5"
    if os.path.exists(BRAIN_MODEL_PATH_H5):
        brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH_H5)
        print("✅ Brain tumor model loaded successfully.")
    else:
        print("⚠️  Brain model file not found. Using mock predictions.")
except ImportError:
    print("⚠️  TensorFlow not installed. Using mock predictions for brain tumor.")
except Exception as e:
    print(f"⚠️  Brain model load error: {e}. Using mock predictions.")

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "sex", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]

FEATURE_LABELS = {
    "age":             "Age",
    "sex":             "Sex",
    "currentSmoker":   "Current Smoker",
    "cigsPerDay":      "Cigarettes Per Day",
    "BPMeds":          "On BP Medications",
    "prevalentStroke": "History of Stroke",
    "prevalentHyp":    "Prevalent Hypertension",
    "diabetes":        "Diabetes",
    "totChol":         "Total Cholesterol",
    "sysBP":           "Systolic Blood Pressure",
    "diaBP":           "Diastolic Blood Pressure",
    "BMI":             "BMI",
    "heartRate":       "Resting Heart Rate",
    "glucose":         "Glucose Level",
}

BRAIN_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# ── Heart: Recommendations & Diet ───────────────────────────────────────────
def get_recommendations(prediction, inputs):
    recs = []
    age            = inputs.get("age",            0)
    currentSmoker  = inputs.get("currentSmoker",  0)
    cigsPerDay     = inputs.get("cigsPerDay",      0)
    BPMeds         = inputs.get("BPMeds",          0)
    prevalentStroke= inputs.get("prevalentStroke", 0)
    prevalentHyp   = inputs.get("prevalentHyp",   0)
    diabetes       = inputs.get("diabetes",        0)
    totChol        = inputs.get("totChol",         0)
    sysBP          = inputs.get("sysBP",           0)
    diaBP          = inputs.get("diaBP",           0)
    BMI            = inputs.get("BMI",             0)
    heartRate      = inputs.get("heartRate",       0)
    glucose        = inputs.get("glucose",         0)

    if prediction == 1:
        recs.append("⚠️ Elevated 10-year heart disease risk detected — consult a cardiologist promptly.")
        recs.append("📉 Monitor blood pressure and cholesterol levels at least every 3 months.")
        recs.append("💊 Discuss preventive medication (statins, antihypertensives) with your doctor.")
        recs.append("🏃 Adopt a structured cardiac rehabilitation exercise program.")
    else:
        recs.append("✅ Low 10-year heart disease risk — keep up your healthy habits.")
        recs.append("🏃 Maintain regular physical activity (at least 150 min/week of moderate exercise).")
        recs.append("🩺 Schedule an annual wellness check to track key cardiac markers.")

    # Smoking
    if currentSmoker == 1:
        recs.append("🚭 Smoking significantly raises cardiac risk — seek cessation support immediately.")
        if cigsPerDay > 20:
            recs.append("🚬 Heavy smoker (>20 cigs/day) — nicotine replacement therapy or varenicline is strongly advised.")

    # Cholesterol
    if totChol > 240:
        recs.append("🥗 High cholesterol (>240 mg/dL) — reduce saturated fats, fried foods, and consider statin therapy.")
    elif totChol > 200:
        recs.append("🥗 Borderline cholesterol — limit full-fat dairy, processed meats, and increase dietary fiber.")

    # Blood pressure
    if sysBP > 140 or diaBP > 90:
        recs.append("🧂 Hypertension detected — reduce sodium intake to <2g/day and follow up with your doctor.")
    elif sysBP > 130:
        recs.append("🧂 Elevated blood pressure — limit salt, alcohol, and increase potassium-rich foods.")

    # Diabetes & glucose
    if diabetes == 1:
        recs.append("🩸 Diabetic patients have 2x higher cardiac risk — maintain strict glucose control (HbA1c < 7%).")
    elif glucose > 126:
        recs.append("🩸 Fasting glucose above 126 mg/dL suggests diabetes — consult your doctor for HbA1c testing.")
    elif glucose > 100:
        recs.append("🩸 Pre-diabetic glucose range — adopt a low-glycaemic diet and increase physical activity.")

    # BMI
    if BMI >= 30:
        recs.append("⚖️ Obesity (BMI ≥ 30) increases cardiac load — aim for a 5–10% body weight reduction.")
    elif BMI >= 25:
        recs.append("⚖️ Overweight (BMI 25–29.9) — a calorie-controlled diet and daily activity can reduce risk.")

    # Heart rate
    if heartRate > 100:
        recs.append("💓 Resting heart rate above 100 bpm (tachycardia) — consult a doctor for evaluation.")
    elif heartRate < 50:
        recs.append("💓 Very low resting heart rate — monitor for symptoms like dizziness or fatigue.")

    # History
    if prevalentStroke == 1:
        recs.append("🧠 Prior stroke history significantly elevates cardiac risk — adhere strictly to antiplatelet therapy.")
    if prevalentHyp == 1 and BPMeds == 0:
        recs.append("💊 Hypertension diagnosed but no BP medication noted — discuss treatment options with your doctor.")
    if BPMeds == 1:
        recs.append("💊 Continue BP medications as prescribed and avoid abrupt discontinuation.")

    # Age
    if age > 60:
        recs.append("🩺 Patients over 60 should have annual cardiac screenings including ECG and lipid panels.")
    elif age > 45:
        recs.append("🩺 Biannual cardiac check-ups are recommended for your age group.")

    return recs


def get_diet_plan(prediction, inputs):
    totChol   = inputs.get("totChol",  0)
    sysBP     = inputs.get("sysBP",    0)
    diabetes  = inputs.get("diabetes", 0)
    BMI       = inputs.get("BMI",      0)
    glucose   = inputs.get("glucose",  0)
    smoker    = inputs.get("currentSmoker", 0)

    avoid = ["Excess sugar and sweets", "Trans fats and hydrogenated oils", "Ultra-processed packaged foods"]

    if totChol > 200:
        avoid.append("Saturated fats: red meat, full-fat dairy, butter")
    if sysBP > 130:
        avoid.append("High-sodium foods: pickles, papad, canned soups, salty snacks")
    if diabetes == 1 or glucose > 100:
        avoid.append("Refined carbohydrates: white bread, white rice, sugary drinks")
    if BMI >= 25:
        avoid.append("Deep-fried foods and calorie-dense fast food")
    if smoker == 1:
        avoid.append("Alcohol — it compounds cardiovascular risk in smokers")

    if prediction == 1:
        return {
            "Morning":     [
                "Oatmeal with flaxseeds, chia seeds and mixed berries",
                "1 glass warm lemon water or green tea (no sugar)",
                "4–5 soaked almonds or walnuts",
            ],
            "Mid-Morning": [
                "1 fresh fruit (apple, pear or guava — low glycaemic)",
                "A small handful of pumpkin or sunflower seeds",
            ],
            "Lunch":       [
                "Grilled fish (salmon/mackerel) or paneer with steamed vegetables",
                "Brown rice or quinoa (small portion — 1 katori)",
                "Mixed green salad with olive oil & lemon dressing",
                "1 bowl of dal or legume soup",
            ],
            "Evening":     [
                "Chamomile or hibiscus herbal tea (cardio-protective)",
                "Whole-grain crackers with hummus or avocado",
            ],
            "Dinner":      [
                "Vegetable stew or lentil soup (low sodium)",
                "Steamed broccoli, spinach or kale",
                "1–2 whole wheat rotis (no butter)",
                "Eat at least 2 hours before sleeping",
            ],
            "Avoid":       avoid,
        }
    else:
        plan = {
            "Morning":     [
                "Whole grain toast with natural peanut butter",
                "Fresh fruit smoothie (banana, berries, low-fat milk or soy milk)",
                "1 boiled egg or a small bowl of Greek yogurt",
            ],
            "Mid-Morning": [
                "Mixed nuts (unsalted: almonds, cashews, walnuts)",
                "1 fresh seasonal fruit",
            ],
            "Lunch":       [
                "Grilled chicken, fish or chickpeas (plant-based protein)",
                "Whole wheat roti or brown rice",
                "Vegetable curry, stir-fry or dal",
                "Cucumber, tomato and onion salad",
            ],
            "Evening":     [
                "Green tea or black coffee (no sugar)",
                "Roasted makhana (fox nuts) or unsalted popcorn",
            ],
            "Dinner":      [
                "Dal (lentils) with mixed vegetables",
                "2 whole wheat rotis",
                "Light vegetable soup or raita (low-fat yogurt with cucumber)",
            ],
            "Avoid":       avoid,
        }
        return plan


# ── Brain: Recommendations & Diet ───────────────────────────────────────────
def get_brain_recommendations(tumor_class):
    base = {
        "Glioma": [
            "⚠️ Glioma detected — consult a neurosurgeon immediately.",
            "🏥 MRI with contrast and biopsy may be required for staging.",
            "💊 Discuss chemotherapy and radiation therapy options with your oncologist.",
            "🧘 Consider psychological support and stress management techniques.",
            "🩺 Schedule follow-up scans every 3 months as advised by your doctor.",
        ],
        "Meningioma": [
            "⚠️ Meningioma detected — seek evaluation from a neurologist or neurosurgeon.",
            "🔍 Many meningiomas are slow-growing; watchful waiting may be appropriate.",
            "🏥 Surgical resection or radiation may be recommended based on size and location.",
            "📋 Regular MRI monitoring every 6–12 months is essential.",
            "🧘 Manage headaches and symptoms with guidance from your healthcare provider.",
        ],
        "Pituitary": [
            "⚠️ Pituitary tumor detected — consult an endocrinologist promptly.",
            "🔬 Hormonal blood tests are required to assess pituitary function.",
            "💊 Medical management (e.g., dopamine agonists) may be sufficient for some tumors.",
            "🏥 Transsphenoidal surgery is an option for larger or symptomatic tumors.",
            "👁️ Regular vision and hormonal monitoring is strongly advised.",
        ],
        "No Tumor": [
            "✅ No tumor detected — your MRI appears normal.",
            "🏃 Maintain a healthy lifestyle with regular exercise and balanced diet.",
            "🧘 Practice stress reduction and ensure adequate sleep (7–9 hrs/night).",
            "🩺 Annual neurological check-ups are recommended if symptoms persist.",
            "💧 Stay well hydrated and limit alcohol and smoking.",
        ],
    }
    return base.get(tumor_class, ["🩺 Consult a medical specialist for further evaluation."])


def get_brain_diet_plan(tumor_class):
    if tumor_class == "No Tumor":
        return {
            "Morning":     ["Whole grain cereal or oats with berries", "Green tea or black coffee", "1 boiled egg or yogurt"],
            "Mid-Morning": ["Handful of almonds or walnuts", "A fresh fruit"],
            "Lunch":       ["Grilled fish or lentils", "Brown rice or whole wheat roti", "Mixed vegetable salad"],
            "Evening":     ["Herbal tea", "Roasted seeds or light snack"],
            "Dinner":      ["Vegetable soup or dal", "Steamed greens", "Light grain portion"],
            "Avoid":       ["Processed and packaged foods", "Excess sugar", "Trans fats"],
        }
    else:
        return {
            "Morning":     ["Turmeric milk or green smoothie", "Oatmeal with flaxseeds and blueberries", "Green tea (anti-inflammatory)"],
            "Mid-Morning": ["Walnuts, almonds (omega-3 rich)", "Fresh pomegranate or berries"],
            "Lunch":       ["Grilled salmon or tofu (anti-inflammatory)", "Quinoa or brown rice", "Steamed broccoli, spinach, kale"],
            "Evening":     ["Ginger-turmeric herbal tea", "Sunflower or pumpkin seeds"],
            "Dinner":      ["Lentil or vegetable soup", "Steamed leafy greens", "Small portion of whole grains"],
            "Avoid":       ["Processed meats and red meat", "Refined sugar and sweets", "Alcohol and smoking", "Fried and fast foods", "Artificial additives and preservatives"],
        }


# ── Mock Predictors ──────────────────────────────────────────────────────────
def mock_heart_predict(inputs):
    """
    Rule-based mock predictor using Framingham risk factors.
    Predicts disease only when multiple strong risk factors co-exist.
    """
    age            = inputs.get("age",            0)
    currentSmoker  = inputs.get("currentSmoker",  0)
    cigsPerDay     = inputs.get("cigsPerDay",      0)
    BPMeds         = inputs.get("BPMeds",          0)
    prevalentStroke= inputs.get("prevalentStroke", 0)
    prevalentHyp   = inputs.get("prevalentHyp",   0)
    diabetes       = inputs.get("diabetes",        0)
    totChol        = inputs.get("totChol",         0)
    sysBP          = inputs.get("sysBP",           0)
    diaBP          = inputs.get("diaBP",           0)
    BMI            = inputs.get("BMI",             0)
    heartRate      = inputs.get("heartRate",       0)
    glucose        = inputs.get("glucose",         0)

    score = 0.0

    # Age — significant risk after 45
    if age > 65:    score += 0.28
    elif age > 55:  score += 0.18
    elif age > 45:  score += 0.08

    # Smoking
    if currentSmoker == 1:
        score += 0.15
        if cigsPerDay > 20: score += 0.08

    # Cholesterol
    if totChol > 280:   score += 0.18
    elif totChol > 240: score += 0.10
    elif totChol > 200: score += 0.03

    # Systolic BP
    if sysBP > 160:   score += 0.20
    elif sysBP > 140: score += 0.12
    elif sysBP > 130: score += 0.05

    # Diastolic BP
    if diaBP > 100:  score += 0.10
    elif diaBP > 90: score += 0.05

    # Diabetes / Glucose
    if diabetes == 1:   score += 0.18
    elif glucose > 126: score += 0.12
    elif glucose > 100: score += 0.04

    # BMI
    if BMI >= 35:   score += 0.12
    elif BMI >= 30: score += 0.07
    elif BMI >= 25: score += 0.02

    # History / medications
    if prevalentStroke == 1: score += 0.20
    if prevalentHyp == 1:    score += 0.10
    if BPMeds == 1:          score += 0.06  # on meds = managed, slight risk acknowledged

    # Heart rate
    if heartRate > 100: score += 0.08
    elif heartRate < 50: score += 0.05

    # Threshold: need multiple strong risk factors
    prediction = 1 if score >= 0.50 else 0

    if prediction == 1:
        confidence = min(0.93, max(0.55, 0.50 + score * 0.38))
    else:
        confidence = min(0.93, max(0.55, 0.95 - score * 0.85))

    return prediction, confidence


def is_valid_mri(img_array_gray):
    """
    Basic sanity checks to reject non-MRI images.
    Brain MRI scans are typically:
      - Mostly dark (large black background)
      - Grayscale with a bright central region
      - Low color saturation (not a colorful photo)
    Returns (is_valid: bool, reason: str)
    """
    import numpy as np

    # img_array_gray: H x W numpy array, values 0-255
    h, w = img_array_gray.shape

    # 1. Check that at least 20% of pixels are very dark (MRI background)
    dark_ratio = np.sum(img_array_gray < 15) / (h * w)
    if dark_ratio < 0.15:
        return False, "Image does not appear to be a brain MRI scan (no dark background detected). Please upload a grayscale MRI image."

    # 2. Check that there IS a bright region (brain tissue)
    bright_ratio = np.sum(img_array_gray > 80) / (h * w)
    if bright_ratio < 0.05:
        return False, "Image appears too dark or blank. Please upload a valid brain MRI scan."

    # 3. Reject if the image is almost entirely one flat color (solid color images)
    std_dev = float(np.std(img_array_gray))
    if std_dev < 8:
        return False, "Image has no variation — please upload a real brain MRI scan."

    return True, "ok"


def check_color_saturation(pil_img_rgb):
    """
    Reject obviously colorful photos (selfies, food, animals, etc.)
    Real MRI scans are near-grayscale even if saved as RGB.
    Returns (is_grayscale_enough: bool, reason: str)
    """
    import numpy as np
    arr = np.array(pil_img_rgb, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Max channel difference per pixel — low for grayscale images
    max_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    if max_diff > 30:
        return False, "This looks like a color photograph, not an MRI scan. Please upload a brain MRI image (grayscale scan)."
    return True, "ok"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/brain")
def brain():
    return render_template("brain.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        input_df = pd.DataFrame([{
            "age":             float(data["age"]),
            "sex":             float(data["sex"]),
            "currentSmoker":   float(data["currentSmoker"]),
            "cigsPerDay":      float(data["cigsPerDay"]),
            "BPMeds":          float(data["BPMeds"]),
            "prevalentStroke": float(data["prevalentStroke"]),
            "prevalentHyp":    float(data["prevalentHyp"]),
            "diabetes":        float(data["diabetes"]),
            "totChol":         float(data["totChol"]),
            "sysBP":           float(data["sysBP"]),
            "diaBP":           float(data["diaBP"]),
            "BMI":             float(data["BMI"]),
            "heartRate":       float(data["heartRate"]),
            "glucose":         float(data["glucose"]),
        }])

        if heart_model is not None:
            prediction  = int(heart_model.predict(input_df)[0])
            proba       = heart_model.predict_proba(input_df)[0]
            confidence  = float(proba[prediction])
            rf          = heart_model.named_steps["model"]
            importances = rf.feature_importances_
            print("=" * 50)
            print("🔴 REAL MODEL used")
            print(f"   Input:      {dict(zip(FEATURE_NAMES, input_df.values[0]))}")
            print(f"   Proba:      No Disease={proba[0]:.3f}  Disease={proba[1]:.3f}")
            print(f"   Prediction: {prediction} ('Disease' if prediction==1 else 'No Disease')")
            print("=" * 50)
        else:
            prediction, confidence = mock_heart_predict(data)
            importances = [0.12, 0.06, 0.10, 0.07, 0.05, 0.08, 0.09, 0.07, 0.10, 0.11, 0.06, 0.05, 0.02, 0.02]
            print("=" * 50)
            print("🟡 MOCK predictor used (no real model found)")
            print(f"   Input:      age={data.get('age')} sysBP={data.get('sysBP')} totChol={data.get('totChol')} glucose={data.get('glucose')}")
            print(f"   Prediction: {prediction} ('Disease' if prediction==1 else 'No Disease')  confidence={confidence:.3f}")
            print("=" * 50)

        feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:5]
        top_features = [
            {"feature": FEATURE_LABELS[f], "importance": round(float(v) * 100, 1)}
            for f, v in feat_imp
        ]

        inputs = {k: float(v) for k, v in data.items() if k in FEATURE_NAMES}
        recommendations = get_recommendations(prediction, inputs)
        diet_plan       = get_diet_plan(prediction, inputs)

        return jsonify({
            "prediction":    prediction,
            "label":         "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "confidence":    round(confidence * 100, 1),
            "recommendations": recommendations,
            "diet_plan":     diet_plan,
            "top_features":  top_features,
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict_brain", methods=["POST"])
def predict_brain():
    try:
        from PIL import Image
        import io

        if "file" not in request.files:
            return jsonify({"error": "No image file uploaded. Use field name 'file'."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        # ── Allowed file extensions ──────────────────────────────────────────
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type '{ext}'. Please upload a JPG, PNG, or WEBP image."}), 400

        # ── Read image ───────────────────────────────────────────────────────
        img_bytes = file.read()
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.verify()  # Check it's a real image, not a renamed file
            pil_img = Image.open(io.BytesIO(img_bytes))  # Re-open after verify
        except Exception:
            return jsonify({"error": "File could not be read as an image. Please upload a valid MRI scan."}), 400

        # ── Color saturation check (reject colorful photos) ──────────────────
        rgb_img = pil_img.convert("RGB")
        color_ok, color_reason = check_color_saturation(rgb_img)
        if not color_ok:
            return jsonify({"error": color_reason}), 400

        # ── Grayscale MRI structure check ────────────────────────────────────
        gray_img = pil_img.convert("L")
        gray_arr = np.array(gray_img)
        mri_ok, mri_reason = is_valid_mri(gray_arr)
        if not mri_ok:
            return jsonify({"error": mri_reason}), 400

        # ── Block predictions if no real model is loaded ─────────────────────
        if brain_model is None:
            return jsonify({
                "error": (
                    "No brain tumor model is loaded on the server. "
                    "Please place 'brain_tumor_model.h5' in the project folder and restart the app. "
                    "Random predictions are disabled to prevent misleading results."
                )
            }), 503

        # ── Run model ────────────────────────────────────────────────────────
        try:
            IMG_SIZE = (224, 224)
            img_resized = rgb_img.resize(IMG_SIZE)
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            raw_probs = brain_model.predict(arr)[0]
            probs     = [round(float(p), 4) for p in raw_probs]
            pred_idx  = int(np.argmax(probs))

        except Exception as e:
            return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

        tumor_class = BRAIN_CLASSES[pred_idx]
        confidence  = round(probs[pred_idx] * 100, 1)
        is_tumor    = tumor_class != "No Tumor"

        class_probabilities = {
            cls: round(probs[i] * 100, 1) for i, cls in enumerate(BRAIN_CLASSES)
        }

        recommendations = get_brain_recommendations(tumor_class)
        diet_plan       = get_brain_diet_plan(tumor_class)

        return jsonify({
            "prediction":          1 if is_tumor else 0,
            "result":              tumor_class,
            "label":               tumor_class,
            "confidence":          confidence,
            "class_probabilities": class_probabilities,
            "recommendations":     recommendations,
            "diet_plan":           diet_plan,
        })

    except Exception as e:
        return jsonify({"error": f"Brain prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
