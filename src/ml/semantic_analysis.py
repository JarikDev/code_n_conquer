import os
import re
import json

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy2

# Создание папки для модели
os.makedirs("model", exist_ok=True)

# 1. Загрузка данных
dfs = []
for i in range(1, 7):
    df = pd.read_csv(f"data/{i}.csv")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# 2. Очистка текста и лемматизация
morph = pymorphy2.MorphAnalyzer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)  # Убираем все кроме букв и пробелов
    text = re.sub(r"\s+", " ", text)  # Убираем лишние пробелы

    # Лемматизация
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]

    return " ".join(lemmatized_words)

# Применение очистки текста
for col in ['doc_text', 'image2text', 'speech2text']:
    data[col] = data[col].apply(clean_text)

# 3. Создание нового столбца с объединенными текстами
data['full_text'] = data['doc_text'] + " " + data['image2text'] + " " + data['speech2text']

# 4. Загрузка модели BERT и токенизатора
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

# 5. Функция для получения эмбеддингов текста
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        # Токенизация текста
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            output = model(**inputs)
        # Получаем эмбеддинг для [CLS] токена (первый токен)
        embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# 6. Получение эмбеддингов для документов
doc_texts = data['full_text'].tolist()
doc_embeddings = get_embeddings(doc_texts)

# 7. Генерация тестового набора для оценки
test_queries = [
    {"query": "погода на завтра", "relevant_ids": [0, 3]},
    {"query": "новости спорта", "relevant_ids": [5, 12]},
    {"query": "разговор о здоровье", "relevant_ids": [8]},
]

# 8. Вычисление метрик поиска
from sklearn.metrics import f1_score

def compute_search_metrics(queries, embeddings, texts, model, k=5):
    precision_list = []
    recall_list = []
    reciprocal_ranks = []
    f1_list = []

    for item in queries:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        query_clean = clean_text(query)
        query_emb = get_embeddings([query_clean])[0]
        sims = cosine_similarity([query_emb], embeddings)[0]

        top_k_idx = np.argsort(sims)[::-1][:k]

        hits = sum(1 for idx in top_k_idx if idx in relevant_ids)
        precision_at_k = hits / k
        recall_at_k = hits / len(relevant_ids)

        precision_list.append(precision_at_k)
        recall_list.append(recall_at_k)

        rr = 0
        for rank, idx in enumerate(top_k_idx, start=1):
            if idx in relevant_ids:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

        f1 = f1_score([1 if idx in relevant_ids else 0 for idx in top_k_idx], [1] * k)
        f1_list.append(f1)

    metrics = {
        "precision@5": round(np.mean(precision_list), 3),
        "recall@5": round(np.mean(recall_list), 3),
        "mrr": round(np.mean(reciprocal_ranks), 3),
        "f1_score": round(np.mean(f1_list), 3),
        "test_size": len(queries)
    }
    return metrics

search_metrics = compute_search_metrics(test_queries, doc_embeddings, doc_texts, model)

# 9. Сохранение всех моделей и метрик
np.save("model/doc_embeddings.npy", doc_embeddings)
joblib.dump(model, "model/model.pkl")
data[['full_text']].to_csv("model/doc_texts.csv", index=False)

with open("model/search_metrics.json", "w", encoding="utf-8") as f:
    json.dump(search_metrics, f, ensure_ascii=False, indent=2)

print(f"✅ Модель, эмбеддинги, тексты и метрики поиска сохранены в папке 'model/'")
print("📊 Search Metrics:", search_metrics)
