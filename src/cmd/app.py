from flask import Flask, request, render_template
import torch
import joblib
import numpy as np
import pandas as pd
import json
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

# Инициализация Flask
app = Flask(__name__)
base_path = "../ml/model/"

# Загрузка моделей и данных
tokenizer = BertTokenizer.from_pretrained("../ml/model/")  # Путь к директории с моделью
model = BertModel.from_pretrained("../ml/model/")  # Путь к директории с моделью

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
    phrase_clean = clean_text(phrase)

    # Токенизация текста и фразы
    inputs_text = tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs_phrase = tokenizer(phrase_clean, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        # Получаем эмбеддинги для текста и фразы
        text_emb = model(**inputs_text).last_hidden_state.mean(dim=1).squeeze()
        phrase_emb = model(**inputs_phrase).last_hidden_state.mean(dim=1).squeeze()

    # Вычисление сходства косинуса между фразой и текстом
    sim = cosine_similarity([text_emb.numpy()], [phrase_emb.numpy()])[0][0]

    if sim >= 0.6:
        start_pos = text_clean.find(phrase_clean)
        if start_pos != -1:
            end_pos = start_pos + len(phrase_clean)
            position = f"{start_pos}-{end_pos}"
        else:
            position = "примерное совпадение (нет точной позиции)"
        result = f"Позиция: {position} | Вероятность: {round(float(sim), 3)}"
    else:
        result = f"Фраза не найдена. Вероятность: {round(float(sim), 3)}"

    return render_template("index.html", result=result, text=text, phrase=phrase)

@app.route("/metrics")
def metrics():
    return render_template("metrics.html", search_metrics=search_metrics)

if __name__ == "__main__":
    app.run(debug=True, port=8081)
