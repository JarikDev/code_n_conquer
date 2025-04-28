import joblib
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer

from src.ml.semantic_analysis import semantic_search_in_document

# Инициализация Flask
app = Flask(__name__)
base_path = "../ml/model/"

# Загрузка модели sentence-transformers
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Загрузка сохраненных результатов
search_results = joblib.load(f"{base_path}search_results.pkl")


# Главная страница
@app.route("/")
def index():
    return render_template("index.html")


# Обработка запроса для поиска
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    phrase = request.form.get("phrase", "").strip()

    if not text or not phrase:
        return render_template("index.html", result="Введите текст и фразу.", text=text, phrase=phrase)

    result = semantic_search_in_document(text, phrase, model)

    # Отображаем результат
    result_display = f"Позиция: {result['positions']} | Вероятность: {round(result['score'], 3)}"

    return render_template("index.html", result=result_display, text=text, phrase=phrase)


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", search_metrics=search_results)


if __name__ == "__main__":
    app.run(debug=True, port=8081)
