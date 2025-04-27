from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import json
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация Flask
app = Flask(__name__)
base_path = "../ml/model/"
# Загрузка моделей и данных
model = joblib.load(f"{base_path}model.pkl")
doc_embeddings = np.load(f"{base_path}doc_embeddings.npy")
doc_texts = pd.read_csv(f"{base_path}doc_texts.csv")['full_text'].tolist()

# Загрузка метрик
if os.path.exists(f"{base_path}search_metrics.json"):
    with open(f"{base_path}search_metrics.json", "r", encoding="utf-8") as f:
        search_metrics = json.load(f)
else:
    search_metrics = {}

# Очистка текста
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    phrase = request.form.get("phrase", "").strip()

    if not text or not phrase:
        return render_template("index.html", result="Введите текст и фразу.", text=text, phrase=phrase)

    text_clean = clean_text(text)
    text_emb = model.encode([text_clean])

    phrase_clean = clean_text(phrase)
    phrase_emb = model.encode([phrase_clean])

    sim = cosine_similarity(phrase_emb, text_emb)[0][0]

    if sim >= 0.4:
        start_pos = text_clean.find(phrase_clean)
        if start_pos != -1:
            end_pos = start_pos + len(phrase_clean)
            position = f"Точное совпадение: {start_pos}-{end_pos}"
        else:
            position = "Примерное совпадение, точная позиция не найдена."
        result = f"Позиция: {position} | Вероятность: {round(float(sim), 3)}"
    else:
        result = f"Фраза не найдена. Вероятность: {round(float(sim), 3)}"

    return render_template("index.html", result=result, text=text, phrase=phrase)

@app.route("/metrics")
def metrics():
    return render_template("metrics.html", search_metrics=search_metrics)

if __name__ == "__main__":
    app.run(debug=True, port=8081)
