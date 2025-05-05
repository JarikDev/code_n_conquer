import os
import re
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Отключаем TensorFlow (работаем через PyTorch)
os.environ["USE_TF"] = "0"

# === Загрузка или инициализация модели ===
model_path = 'model/semantic_search_model'
if os.path.exists(model_path):
    ST_model = SentenceTransformer(model_path)
    print("✅ Модель загружена локально")
else:
    ST_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    os.makedirs('model', exist_ok=True)
    ST_model.save(model_path)
    print("📥 Модель скачана и сохранена")


# === Очистка текста ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# === Функция поиска ===
def semantic_search_in_document(document, query, model):
    doc_cleaned = clean_text(document)
    query_cleaned = clean_text(query)

    doc_words = doc_cleaned.split()
    query_words = query_cleaned.split()

    if len(query_words) > 2:
        return {"document": document, "distance": 1.0, "positions": "0-0", "matched_word": ""}

    doc_embedding = model.encode(doc_cleaned, convert_to_tensor=True)
    query_embedding = model.encode(query_cleaned, convert_to_tensor=True)

    cosine_similarity = util.cos_sim(query_embedding, doc_embedding)[0]
    cosine_distance = 1.0 - cosine_similarity

    if cosine_distance > 0.5:
        return {"document": document, "distance": float(cosine_distance), "positions": "0-0", "matched_word": ""}

    doc_words_embeddings = model.encode(doc_words, convert_to_tensor=True)
    word_cosine_scores = util.cos_sim(query_embedding, doc_words_embeddings)[0]
    best_word_idx = np.argmax(word_cosine_scores)
    best_word = doc_words[best_word_idx]

    start_pos = document.lower().find(best_word)
    end_pos = start_pos + len(best_word) if start_pos != -1 else 0
    positions_str = f"{start_pos}-{end_pos}" if start_pos != -1 else "0-0"

    return {
        "document": document,
        "distance": float(cosine_distance),
        "positions": positions_str,
        "matched_word": best_word
    }


# === Входные данные ===
documents = [
    'доченька твоя совсем большая стала',
    'в лесу растёт высокое дерево',
    'портфель лежит в шкафу дома',
    'институт находится в центре города',
    'вся дорога забита деревьями и цветами',
    'в следующее воскресенье я собираюсь в питер',
    'у меня сломалась стиралка прикинь',
    'садись в машину и поехали уже',
    'сколько стоит ремонт стиральной машины',
    'ты возьми корзину прежде чем набрать продукты',
    'его сегодня утром отвезли в ближайший госпиталь'
]

queries = [
    'дочь',
    'дерево',
    'портфель',
    'институт',
    'дерево',
    'санкт петербург',
    'стиральная машина',
    'автомобиль',
    'автомобиль',
    'звонить',
    'больница'
]

ground_truth_words = [
    'доченька',
    'дерево',
    'портфель',
    'институт',
    'деревьями',
    'питер',
    'стиралка',
    'машину',
    'машины',
    'корзину',
    'госпиталь'
]

assert len(documents) == len(queries) == len(ground_truth_words), "❌ Длины списков не совпадают"

# === Поиск + метрики ===
results = []
exact_match_count = 0
distances = []

for doc, query, expected_word in zip(documents, queries, ground_truth_words):
    result = semantic_search_in_document(doc, query, ST_model)

    if result["distance"] <= 0.5:
        matched_word = result["matched_word"]
        is_exact = matched_word.lower() == expected_word.lower()
        exact_match_count += int(is_exact)

        results.append({
            "документ": doc,
            "запрос": query,
            "ожидалось": expected_word,
            "найдено": matched_word,
            "позиция": result["positions"],
            "косинусное_расстояние": round(result["distance"], 4),
            "совпадение": is_exact
        })
        distances.append(result["distance"])

# === Вычисление метрик ===
coverage = len(results) / len(documents)
mean_distance = np.mean(distances) if distances else 1.0
accuracy = exact_match_count / len(results) if results else 0.0

metrics = {
    "total_documents": len(documents),
    "matched_documents": len(results),
    "coverage": round(coverage, 3),
    "mean_distance": round(mean_distance, 4),
    "exact_match_accuracy": round(accuracy, 3)
}

# === Вывод результатов ===
print("\n📊 Результаты поиска:")
for r in results:
    print(
        f"📄: {r['документ']}\n🔎: {r['запрос']} → {r['найдено']} | ожидалось: {r['ожидалось']} | позиция: {r['позиция']} | расстояние: {r['косинусное_расстояние']}\n"
    )

print("=== 📈 Метрики ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# === Сохранение результатов ===
os.makedirs("model", exist_ok=True)
joblib.dump(results, "model/search_results.pkl")
joblib.dump(metrics, "model/search_metrics.pkl")
ST_model.save("model/semantic_search_model")
