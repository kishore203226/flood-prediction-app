from flask import Flask, render_template_string, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load("flood_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
except Exception as e:
    raise Exception(f"Model or Scaler file not found: {e}")


HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flood Prediction System</title>

<style>
body {
    margin: 0;
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.navbar {
    background: rgba(0, 0, 0, 0.3);
    padding: 15px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    backdrop-filter: blur(10px);
}

.container {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}

.card {
    background: rgba(255, 255, 255, 0.15);
    padding: 40px;
    border-radius: 15px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    width: 400px;
    color: white;
    animation: fadeIn 1s ease-in-out;
}

input {
    width: 100%;
    padding: 10px;
    margin: 8px 0;
    border: none;
    border-radius: 8px;
}

button {
    width: 100%;
    padding: 12px;
    margin-top: 15px;
    border: none;
    border-radius: 8px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 16px;
    cursor: pointer;
    transition: 0.3s ease;
}

button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00c6ff;
}

.result {
    margin-top: 20px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    color: #ffd700;
}

footer {
    text-align: center;
    padding: 15px;
    background: rgba(0, 0, 0, 0.3);
    color: white;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
</head>

<body>

<div class="navbar">
    <h2>Flood Risk Prediction</h2>
    <span>ML Powered System</span>
</div>

<div class="container">
    <div class="card">
        <h3>Enter Rainfall Details</h3>

        <form method="POST" action="/predict">
            <input type="number" step="any" name="cloud_cover" placeholder="Cloud Cover (%)" required>
            <input type="number" step="any" name="annual" placeholder="Annual Rainfall (mm)" required>
            <input type="number" step="any" name="jan_feb" placeholder="Jan-Feb Rainfall (mm)" required>
            <input type="number" step="any" name="mar_may" placeholder="Mar-May Rainfall (mm)" required>
            <input type="number" step="any" name="jun_sep" placeholder="Jun-Sep Rainfall (mm)" required>

            <button type="submit">Predict Flood Risk</button>
        </form>

        {% if prediction %}
        <div class="result">
            {{ prediction }}
        </div>
        {% endif %}
    </div>
</div>

<footer>
    © 2026 Flood Prediction System | Flask + XGBoost
</footer>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        cloud_cover = float(request.form["cloud_cover"])
        annual = float(request.form["annual"])
        jan_feb = float(request.form["jan_feb"])
        mar_may = float(request.form["mar_may"])
        jun_sep = float(request.form["jun_sep"])

        data = np.array([[cloud_cover, annual, jan_feb, mar_may, jun_sep]])
        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        if prediction == 1:
            result = f"High Flood Risk ({probability*100:.2f}% probability)"
        else:
            result = f"Low Flood Risk ({(1-probability)*100:.2f}% probability)"

        return render_template_string(HTML_FORM, prediction=result)

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
