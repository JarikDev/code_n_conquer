import os

import joblib
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer

from src.ml.ml_utils import search_in_doc

# Инициализация Flask
app = Flask(__name__)
base_path = "../ml/model/"

# === Загрузка модели ===
# Если модель уже сохранена локально — загружаем её
model_path = os.path.join(base_path, "semantic_search_model")

if os.path.exists(model_path):
    model = SentenceTransformer(model_path)
    print("✅ Локальная модель загружена.")
else:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("⚠️ Модель загружена из интернета.")
    os.makedirs(base_path, exist_ok=True)
    model.save(model_path)
    print(f"💾 Модель сохранена в {model_path}")

# === Загрузка предобработанных результатов ===
search_results_path = os.path.join(base_path, "search_results.pkl")
search_results = joblib.load(search_results_path)


# === Главная страница ===
@app.route("/")
def index():
    return render_template("index.html")


# === Обработка поиска ===
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    phrase = request.form.get("phrase", "").strip()

    if not text or not phrase:
        return render_template("index.html", result="Введите текст и фразу.", text=text, phrase=phrase)

    result = search_in_doc(text, phrase, model)

    # Проверим, был ли найден результат
    if result["matched_word"]:
        result_display = (
            f"🔍 Найдено слово: <b>{result['matched_word']}</b><br>"
            f"📍 Позиция: {result['positions']}<br>"
            f"🧠 Косинусное расстояние: {round(result['distance'], 3)}"
        )
    else:
        result_display = "🚫 Ничего не найдено (слишком большое расстояние или неподходящий запрос)."

    return render_template("index.html", result=result_display, text=text, phrase=phrase)


# === Страница с метриками ===
@app.route("/metrics")
def metrics():
    metrics_path = os.path.join(base_path, "search_metrics.pkl")
    if os.path.exists(metrics_path):
        search_metrics = joblib.load(metrics_path)
    else:
        search_metrics = {"error": "Метрики не найдены. Запустите модельное ядро."}

    return render_template(
        "metrics.html",
        search_metrics=search_metrics,
        search_results=search_results
    )


if __name__ == "__main__":
    app.run(debug=True, port=8081)
