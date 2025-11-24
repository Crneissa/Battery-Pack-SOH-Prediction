from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")  # for servers / no GUI
import matplotlib.pyplot as plt

import base64
from io import BytesIO
import json
import re

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
app = Flask(__name__)

API_KEY = "AIzaSyC9ZOUoxihH0RKCRRLJmnaE4klu07v_E3A"  # <-- put your real key here

DATA_FILE = "PulseBat_Dataset.xlsx"
SHEET_NAME = "SOC ALL"
FEATURE_COLS = [f"U{i}" for i in range(1, 22)]

THRESHOLD_DEFAULT = 0.6
MIN_VOLTAGE = 0.0
MAX_VOLTAGE = 5.0

# Gemini Setup
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --------------------------------------------------------
# GEMINI ROUTER PROMPT
# --------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """
You are a router for a battery health prediction system.

Your job is to decide whether the user's message is:
1. A general chat message (normal conversational question)
2. A battery State of Health (SOH) prediction request.

Your output MUST ALWAYS be valid JSON only.

Rules:

IF the user is asking for SOH prediction, battery health, battery condition, voltage evaluation, or provides numbers:
    - Set "route" to "PREDICT".
    - Extract exactly 21 voltage values in numeric form and place them in "values".
    - Values may appear comma-separated, space-separated, mixed, or messy.
    - If fewer than 21 numeric values are detected, put "values": null.

IF the user is NOT asking for prediction:
    - Set "route" to "CHAT".
    - Provide a natural conversational response in "response".

JSON format:
{
  "route": "CHAT" or "PREDICT",
  "values": [list of 21 voltages OR null],
  "response": "chat response ONLY if route=CHAT"
}

DO NOT add explanations, comments, or extra text.
Return ONLY the JSON.
"""

# --------------------------------------------------------
# DATA

FAQ_KNOWLEDGE_BASE = {
    "health_factors": {
        "keywords": ["factor", "health", "affect", "destroy", "damage", "cause"], 
        "required_match_count": 2, 
        "answer": (
            "Here are some **key factors that affect battery health**:\n\n"
            "1. **Charge/Discharge Cycles** – Frequent fast charging or deep discharging (0–100%) accelerates wear.\n"
            "2. **Depth of Discharge (DoD)** – Keeping the battery roughly between **20–80%** is gentler.\n"
            "3. **Temperature** – Heat speeds up chemical ageing.\n"
            "4. **Charging Rate** – Fast charging generates more heat and stress.\n"
            "5. **Storage** – Store at moderate charge (40-60%) and cool temps.\n\n"
            "Your **voltage profile (U1–U21)** helps us estimate the resulting **SOH**."
        )
    },
    "extend_life": {
        "keywords": ["extend", "improve", "increase", "life", "lifespan", "save", "longer"],
        "required_match_count": 2,
        "answer": (
            "Practical ways to **extend your battery's life**:\n\n"
            "1. **Avoid extremes** – Stay between **20–80%** daily.\n"
            "2. **Limit heat** – Don't charge/store in hot environments.\n"
            "3. **Slow Charge** – Use moderate power when you can.\n"
            "4. **Avoid deep discharge** – Charge before you hit 0%.\n"
            "5. **Storage** – Keep at **40–60%** if storing for long periods."
        )
    },
    "what_is_soh": {
        "keywords": ["soh", "state of health", "meaning", "definition", "define"],
        "required_match_count": 1, 
        "answer": (
            "**State of Health (SOH)** describes **how much usable capacity a battery has left**:\n\n"
            "- **SOH = 1.0 (100%)** → Like new.\n"
            "- **SOH = 0.8 (80%)** → 80% capacity remaining.\n"
            "- **< 0.6 (60%)** → Often considered end-of-life.\n\n"
            "We use your **U1–U21 voltages** to predict this value."
        )
    },
    "voltage_range": {
        "keywords": ["voltage", "range", "optimal", "limit", "min", "max", "normal"],
        "required_match_count": 2,
        "answer": (
            "Typical **per-cell voltage ranges** (Li-ion):\n\n"
            "- **Fully charged**: ~4.1–4.2 V\n"
            "- **Nominal**: ~3.6–3.7 V\n"
            "- **Min limit**: ~3.0 V (going lower damages the cell)."
        )
    }
}
# --------------------------------------------------------

# --------------------------------------------------------
# MODEL LOADING & TRAINING
# --------------------------------------------------------

def load_and_train_model():
    """Load dataset, train regression, and compute performance metrics."""
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\(.*\)", "", regex=True)
        .str.replace(r"[_\.\s]+", "", regex=True)
        .str.upper()
    )

    X = df[FEATURE_COLS]
    y = df["SOH"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    # Build regression equation string
    coefs = model.coef_
    intercept = model.intercept_
    equation_parts = [f"{intercept:.4f}"]
    for i, c in enumerate(coefs, start=1):
        sign = "+" if c >= 0 else "-"
        equation_parts.append(f" {sign} {abs(c):.4f}*U{i}")
    regression_equation = "SOH = " + "".join(equation_parts)

    return model, X_train, X_test, y_train, y_test, test_r2, test_mse, test_mae, regression_equation


model, X_train, X_test, y_train, y_test, TEST_R2, TEST_MSE, TEST_MAE, regression_equation = load_and_train_model()


# --------------------------------------------------------
# PLOTTING HELPERS
# --------------------------------------------------------

def generate_performance_plot(current_soh=None):
    """
    Scatter plot of test predictions vs actual.
    Optionally highlight current predicted SOH as a vertical line.
    """
    y_pred = model.predict(X_test)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Test Data")

    # Plot y = x line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")

    if current_soh is not None:
        plt.axvline(current_soh, color="green", linestyle=":", label="Your SOH")

    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Model Performance on Test Set")
    plt.legend()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return img_b64


def generate_soh_bar_plot(values, predicted_soh):
    """Bar plot of U1–U21 voltages."""
    plt.figure(figsize=(10, 4))
    plt.bar([f"U{i}" for i in range(1, 22)], values)
    plt.xticks(rotation=45, fontsize=7)
    plt.ylabel("Voltage (V)")
    plt.title(f"Voltages U1–U21 (Predicted SOH = {predicted_soh:.3f})")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return img_b64


def generate_input_domain_plot(values, soh_clamped):
    """
    Show how the user's input compares to the dataset domain.
    X-axis: mean voltage of each sample (U1–U21)
    Y-axis: actual SOH (from dataset test set)
    User's point: (mean_user_voltage, soh_clamped) as a red X.
    """
    # Mean voltages for test samples
    test_means = X_test.mean(axis=1).values  # shape (n_test,)
    test_soh = y_test.values

    user_mean = np.mean(values)

    plt.figure(figsize=(6, 4))
    plt.scatter(test_means, test_soh, alpha=0.6, label="Test Samples")
    plt.scatter(
        [user_mean],
        [soh_clamped],
        marker="X",
        s=120,
        edgecolors="black",
        linewidths=1.5,
        label="Your Input"
    )

    plt.xlabel("Mean Voltage (U1–U21)")
    plt.ylabel("SOH")
    plt.title("Your Input vs Training Domain")
    plt.legend()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return img_b64


# --------------------------------------------------------
# HELPERS: VALIDATION, SOH PROCESSING
# --------------------------------------------------------

def missing_value_names(values):
    if values is None:
        return list(range(1, 22))
    count = len(values)
    if count >= 21:
        return []
    return list(range(count + 1, 22))


def detect_and_suggest_fixes(values, min_v=MIN_VOLTAGE, max_v=MAX_VOLTAGE):
    issues = []

    for v in values:
        if v < min_v or v > max_v:
            suggestion = None

            # Very big number like 3247 -> 3.247
            if v > 10:
                s = str(int(v))
                if len(s) > 1:
                    suggestion = float(s[0] + "." + s[1:])
            # 10 > v > max_v, e.g., 7.0 -> 0.7; assume decimal shift
            elif max_v < v < 10:
                suggestion = v / 10.0

            # Make sure suggestion is valid
            if suggestion is not None and not (min_v <= suggestion <= max_v):
                suggestion = None

            issues.append({"bad": v, "suggested": suggestion})

    return issues


def run_prediction(values, threshold):
    df_input = pd.DataFrame([values], columns=FEATURE_COLS)
    soh = model.predict(df_input)[0]
    # Classification is done on clamped value, in process_soh_output
    return soh


def process_soh_output(soh_raw, threshold):
    """
    Clamp SOH to [0, 1] and generate:
      - clamped SOH
      - classification based on clamped SOH
      - warning if raw SOH out of known data range
    """
    soh_clamped = max(0.0, min(1.0, soh_raw))

    warning = ""
    if soh_raw != soh_clamped:
        warning = (
            f"⚠️ Raw model output ({soh_raw:.4f}) is outside the valid SOH range "
            f"learned from data (~0.61–0.92). It has been clamped to "
            f"{soh_clamped:.4f}. Inputs may be unrealistic or outside "
            f"the trained operating domain."
        )

    health = "Healthy" if soh_clamped >= threshold else "Unhealthy"
    return soh_clamped, health, warning


def answer_battery_faq(user_msg: str):
    if not user_msg:
        return None
    
    msg = user_msg.lower().strip()
    best_match = None
    highest_score = 0

    for topic, data in FAQ_KNOWLEDGE_BASE.items():
        matches = 0
        for keyword in data["keywords"]:
            if keyword in msg:
                matches +=1
    
    if matches >= data["required_match_count"]:
        if matches > highest_score:
            highest_score = matches
            best_match = data["answer"]

    return best_match
# --------------------------------------------------------
# ROUTES
# --------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = str(data.get("message", "")).strip()
    want_plot = data.get("plot", False)
    threshold = float(data.get("threshold", THRESHOLD_DEFAULT))

    if not user_msg:
        return jsonify({
            "response": "Please enter a message.",
            "type": "error"
        })

    # 0️⃣ FAST PATH: if user typed exactly 21 numbers, treat as SOH input
    num_strings = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", user_msg)
    try:
        values = [float(x) for x in num_strings]
    except ValueError:
        values = []

    if len(values) == 21:
        # --- same behaviour as before for prediction ---
        issues = detect_and_suggest_fixes(values, MIN_VOLTAGE, MAX_VOLTAGE)
        if issues:
            msg_lines = ["**Invalid or unusual voltage values detected:**", ""]
            for item in issues:
                bad = item["bad"]
                suggested = item["suggested"]
                if suggested is not None:
                    msg_lines.append(f"- **{bad}** → did you mean **{suggested}** volts?")
                else:
                    msg_lines.append(
                        f"- **{bad}** is outside the valid range (**{MIN_VOLTAGE}–{MAX_VOLTAGE} V**)"
                    )
            msg_lines.append("")
            msg_lines.append(
                "Please correct these values (or resend all 21 voltages) before I run the prediction.\n"
                "Prediction has been stopped for safety."
            )

            return jsonify({
                "response": "\n".join(msg_lines),
                "type": "prediction_request"
            })

        # Prediction & SOH processing
        soh_raw = run_prediction(values, threshold)
        soh_clamped, health, warning = process_soh_output(soh_raw, threshold)

        soh_plot_img = generate_soh_bar_plot(values, soh_clamped) if want_plot else None
        performance_img = generate_performance_plot(current_soh=soh_clamped) if want_plot else None
        domain_img = generate_input_domain_plot(values, soh_clamped) if want_plot else None
        input_preview = ", ".join(f"{v:.3f}" for v in values[:6])

        result = f"""
**Prediction Result**

**Predicted SOH (raw):** {soh_raw:.4f}  
**Predicted SOH (clamped):** {soh_clamped:.4f} ({soh_clamped*100:.2f}%)
**Classification:** **{health}**
**Threshold Used:** {threshold:.4f} ({threshold*100:.1f}%)

{f"**{warning}**" if warning else ""}

**Model Performance (Original Sorting):**
- **Test R²:** {TEST_R2:.4f}
- **Test MSE:** {TEST_MSE:.4f}
- **Test MAE:** {TEST_MAE:.4f}

**Input Voltages (first 6):** {input_preview}

**Regression Equation:**  
{regression_equation}
"""

        # Only add plots if they were generated
        if want_plot and performance_img:
            result += f"""
<br><br><strong>SOH Prediction Performance</strong><br>
<img src="data:image/png;base64,{performance_img}" alt="SOH Prediction Performance Scatter" />
"""

        if want_plot and domain_img:
            result += f"""
<br><br><strong>Your Input vs Training Domain</strong><br>
<img src="data:image/png;base64,{domain_img}" alt="Input Domain Scatter" />
"""

        return jsonify({
            "response": result,
            "type": "prediction_result",
            "plot": soh_plot_img
        })

    # 1️⃣ NO 21-VALUE INPUT → first check if it's one of the hard-coded FAQ questions
    faq_answer = answer_battery_faq(user_msg)
    if faq_answer is not None:
        return jsonify({
            "response": faq_answer,
            "type": "general"
        })

    # 2️⃣ Anything else → talk to Gemini like a normal chat assistant
    chat_system_prompt = """
You are the AI assistant inside a battery State of Health (SOH) prediction application.

The app has two main functions:
- Predict SOH from 21 voltage measurements (U1–U21) using a regression model.
- Explain battery concepts (SOH, battery health factors, voltage ranges, best practices to extend battery life) to the user.

Guidelines:
- If the user asks a conceptual or practical question about batteries, SOH, or this app, answer clearly and concisely.
- If the user asks how to use the app, explain that they can paste 21 voltage values (U1–U21) and the model will predict SOH and classify the battery based on a threshold.
- Do NOT invent model metrics; the app already displays them.
- Keep answers friendly and focused on batteries / SOH / this tool.
"""

    full_prompt = chat_system_prompt + "\n\nUser:\n" + user_msg + "\n\nAssistant:"

    try:
        gemini_response = gemini_model.generate_content(full_prompt)
        reply_text = gemini_response.text
    except Exception as e:
        print("GEMINI CHAT ERROR:", e)
        reply_text = (
            "I had trouble contacting the AI model.\n\n"
            "You can still enter 21 voltage values (U1–U21) to get an SOH prediction, "
            "or use the quick help buttons for battery health tips."
        )

    return jsonify({
        "response": reply_text,
        "type": "general"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Direct SOH prediction endpoint used by SOH mode in the UI."""
    data = request.json or {}
    raw_values = data.get("values", "")
    threshold = float(data.get("threshold", THRESHOLD_DEFAULT))
    want_plot = data.get("plot", False)  # Get plot preference from request

    # Accept either string "3.56 3.57 ..." or list [3.56, 3.57, ...]
    if isinstance(raw_values, str):
        parts = raw_values.replace(",", " ").split()
        try:
            values = [float(x) for x in parts]
        except ValueError:
            return jsonify({
                "response": (
                    "Some of the values could not be read as numbers.\n\n"
                    "Please resend exactly **21 numeric voltage values (U1–U21)**."
                ),
                "type": "prediction_request"
            })
    elif isinstance(raw_values, list):
        try:
            values = [float(x) for x in raw_values]
        except ValueError:
            return jsonify({
                "response": (
                    "Some of the values could not be read as numbers.\n\n"
                    "Please resend exactly **21 numeric voltage values (U1–U21)**."
                ),
                "type": "prediction_request"
            })
    else:
        return jsonify({
            "response": "Please provide voltage values as text or a list.",
            "type": "prediction_request"
        })

    # 1️⃣ Ensure we have exactly 21 values
    if len(values) != 21:
        missing = missing_value_names(values)
        if len(values) < 21:
            missing_labels = ", ".join([f"U{i}" for i in missing])
            msg = (
                f"You entered **{len(values)} value(s)**, but **21 values** are required.\n\n"
                f"Missing values: **{missing_labels}**\n\n"
                "Please send exactly **21 voltage values (U1–U21)** in order."
            )
        else:
            msg = (
                f"You entered **{len(values)} values**, but **21 values** are required.\n\n"
                "Please send exactly **21 voltage values (U1–U21)**. "
                "Do not include extra numbers."
            )

        return jsonify({
            "response": msg,
            "type": "prediction_request"
        })

    # 2️⃣ Range / sanity checks
    issues = detect_and_suggest_fixes(values, MIN_VOLTAGE, MAX_VOLTAGE)
    if issues:
        msg_lines = ["**Invalid or unusual voltage values detected:**", ""]
        for item in issues:
            bad = item["bad"]
            suggested = item["suggested"]
            if suggested is not None:
                msg_lines.append(f"- **{bad}** → did you mean **{suggested}** volts?")
            else:
                msg_lines.append(
                    f"- **{bad}** is outside the valid range (**{MIN_VOLTAGE}–{MAX_VOLTAGE} V**)"
                )
        msg_lines.append("")
        msg_lines.append(
            "Please correct these values (or resend all 21 voltages) before I run the prediction.\n"
            "Prediction has been stopped for safety."
        )

        return jsonify({
            "response": "\n".join(msg_lines),
            "type": "prediction_request"
        })

    # 3️⃣ Safe to run prediction
    soh_raw = run_prediction(values, threshold)
    soh_clamped, health, warning = process_soh_output(soh_raw, threshold)

    # Plots (only generate if requested)
    soh_plot_img = generate_soh_bar_plot(values, soh_clamped) if want_plot else None
    performance_img = generate_performance_plot(current_soh=soh_clamped) if want_plot else None
    domain_img = generate_input_domain_plot(values, soh_clamped) if want_plot else None
    input_preview = ", ".join(f"{v:.3f}" for v in values[:6])

    result = f"""
**Prediction Result**

**Predicted SOH (raw):** {soh_raw:.4f}  
**Predicted SOH (clamped):** {soh_clamped:.4f} ({soh_clamped*100:.2f}%)
**Classification:** **{health}**
**Threshold Used:** {threshold:.4f} ({threshold*100:.1f}%)

{f"**{warning}**" if warning else ""}

**Model Performance (Original Sorting):**
- **Test R²:** {TEST_R2:.4f}
- **Test MSE:** {TEST_MSE:.4f}
- **Test MAE:** {TEST_MAE:.4f}

**Input Voltages (first 6):** {input_preview}

**Regression Equation:**  
{regression_equation}
"""

    # Only add plots to result if they were generated
    if want_plot and performance_img:
        result += f"""
<br><br><strong>SOH Prediction Performance</strong><br>
<img src="data:image/png;base64,{performance_img}" alt="SOH Prediction Performance Scatter" />
"""

    if want_plot and domain_img:
        result += f"""
<br><br><strong>Your Input vs Training Domain</strong><br>
<img src="data:image/png;base64,{domain_img}" alt="Input Domain Scatter" />
"""

    return jsonify({
        "response": result,
        "type": "prediction_result",
        "plot": soh_plot_img
    })


# --------------------------------------------------------
# RUN APP
# --------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
