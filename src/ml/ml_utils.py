# Загружаем библиотеки
import os
import re

import numpy as np

os.environ["USE_TF"] = "0"

# Для построения модели
from sentence_transformers import util


# Функция для очистки текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Функция для поиска наиболее близкого слова в документе
def search_in_doc(document, query, model):
    """
    Ищет слово в документе, наиболее близкое по смыслу к заданному запросу.

    Args:
        document (str): Строка (документ), в которой выполняется поиск.
        query (str): Запрос (словосочетание) для поиска.
        model: Модель sentence-transformers для создания эмбеддингов.

    Returns:
        dict: Словарь с результатами поиска:
            - "document": Исходный документ.
            - "score": Вероятность совпадения (от 0 до 1).
            - "positions": Позиции найденного слова в формате (start-end) в символах.
            - "matched_word": Найденное слово.
    """

    doc_cleaned = clean_text(document)
    query_cleaned = clean_text(query)

    doc_words = doc_cleaned.split()
    query_words = query_cleaned.split()

    # Проверка на длину запроса
    if len(query_words) > 2:
        return {
            "document": document,
            "distance": 1.0,
            "positions": "0-0",
            "matched_word": ""
        }

    # Получаем эмбеддинги для всего документа и запроса
    doc_embedding = model.encode(doc_cleaned, convert_to_tensor=True)
    query_embedding = model.encode(query_cleaned, convert_to_tensor=True)

    # Вычисляем косинусное сходство между запросом и документом
    cosine_similarity = util.cos_sim(query_embedding, doc_embedding)[0]

    # Косинусное расстояние = 1 - косинусное сходство
    cosine_distance = 1.0 - cosine_similarity

    # Если косинусное расстояние слишком большое (> 0.5), запрос не связан с документом
    if cosine_distance > 0.5:
        return {
            "document": document,
            "distance": float(cosine_distance),
            "positions": "0-0",
            "matched_word": ""
        }

    # Находим ближайшее слово для определения позиции
    doc_words_embeddings = model.encode(doc_words, convert_to_tensor=True)
    word_cosine_scores = util.cos_sim(query_embedding, doc_words_embeddings)[0]
    best_word_idx = np.argmax(word_cosine_scores)
    best_word = doc_words[best_word_idx]

    # Определяем позиции найденного слова
    start_pos = document.lower().find(best_word)
    end_pos = start_pos + len(best_word) if start_pos != -1 else 0
    positions_str = f"{start_pos}-{end_pos}" if start_pos != -1 else "0-0"

    return {
        "document": document,
        "distance": float(cosine_distance),
        "positions": positions_str,
        "matched_word": best_word
    }
