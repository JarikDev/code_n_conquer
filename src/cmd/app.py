from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Загрузка модели и метрик
model = joblib.load("../ml/model/model.pkl")
metrics = joblib.load("../ml/model/metrics.pkl")
vectorizer = joblib.load("../ml/model/vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    if not text:
        return render_template("index.html", result="Введите текст.")

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    print(f'Prediction: {prediction}')
    return render_template("index.html", result=f"Предсказание: {prediction}")

@app.route("/metrics", methods=["GET"])
def get_metrics():
    return render_template("metrics.html", accuracy=metrics["accuracy"], report=metrics["report"])

if __name__ == "__main__":
    app.run(debug=True, port=8081)
