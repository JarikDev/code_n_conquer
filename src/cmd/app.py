from flask import Flask, request, render_template
import joblib
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Инициализация Flask
app = Flask(__name__)
base_path = "../ml/model/"

# Загрузка модели sentence-transformers
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Загрузка сохраненных результатов
search_results = joblib.load(f"{base_path}search_results.pkl")

# Очистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Функция для поиска в документе (для использования в запросах)
def semantic_search_in_document(document, query, model):
    """
    Ищет слово в документе, наиболее близкое по смыслу к заданному запросу.
    """
    doc_cleaned = clean_text(document)
    query_cleaned = clean_text(query)

    # Проверка, что запрос состоит не более чем из 2 слов
    if len(query_cleaned.split()) > 2:
        return {"error": "Запрос должен содержать не более 2 слов."}

    # Разделяем документ на слова
    doc_words = doc_cleaned.split()
    if not doc_words:
        return {'document': document, 'score': 0.0, 'positions': 'Слова не найдены', 'matched_word': ''}

    # Создание эмбеддингов для запроса и слов из документа
    query_embedding = model.encode(query_cleaned, convert_to_tensor=True)
    word_embeddings = model.encode(doc_words, convert_to_tensor=True)

    # Косинусное сходство между запросом и каждым словом в документе
    cosine_scores = util.cos_sim(query_embedding, word_embeddings)[0]

    best_match_idx = None
    best_score = -1

    for i, word in enumerate(doc_words):
        score = cosine_scores[i].item()

        # Подсвечиваем точные совпадения (например, "дочь" в "доченька")
        if query_cleaned in word:
            score += 1.0  # Добавляем бонус

        if score > best_score:
            best_score = score
            best_match_idx = i

    if best_match_idx is None or best_score < 0.05:
        return {'document': document, 'score': 0.0, 'positions': 'Слова не найдены', 'matched_word': ''}

    best_word = doc_words[best_match_idx]
    display_score = 1.0 if best_score >= 0.5 else best_score

    # Вычисляем позиции в символах
    original_words = re.findall(r'\S+', document)
    matched_word = original_words[best_match_idx]

    current_pos = 0
    for i, word in enumerate(original_words):
        start_pos = document.find(word, current_pos)
        end_pos = start_pos + len(word)

        if i == best_match_idx:
            positions_str = f'{start_pos}-{end_pos}'
            break
        current_pos = end_pos

    return {'document': document, 'score': display_score, 'positions': positions_str, 'matched_word': matched_word}

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
