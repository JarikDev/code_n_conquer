# Загружаем библиотеки
import pandas as pd
import numpy as np
import pymorphy3
import re
import joblib
import os
os.environ["USE_TF"] = "0"


# Для построения модели
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util



# Ссылка на raw-версию CSV
url = 'https://raw.githubusercontent.com/JarikDev/code_n_conquer/master/src/ml/data/1.csv'

# Загружаем данные
text_data = pd.read_csv(url)

# Копия датасета
text_df = text_data.copy()



# Выводим информацию о данных
print('Данные имеют следующую размерность: ')
print(f'Количество строк: {text_df.shape[0]}; [Исключённый текст] признаков (столбцов): {text_df.shape[1]}.')
print(' - ' * 40)

# Выводим первые 10 строк
print('\nВыводим первые 10 строк')
print(text_df.head(10))
print('0.0s')



# Загружаем модель sentence-transformers
ST_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('Модель загружена')



# Функция для очистки текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text






# Функция для поиска наиболее близкого слова в документе
def semantic_search_in_document(document, query, model):
    
    
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




# Заданные списки строк (документы) и словосочетаний
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




# Проверяем, что длины списков совпадают
assert len(documents) == len(queries), "Длина списков documents и queries должна совпадать!"




# Выполняем поиск
results = []
for doc, query in zip(documents, queries):
    result = semantic_search_in_document(doc, query, ST_model)
    
    # Добавляем результат только если косинусное расстояние <= 0.5
    if result["distance"] <= 0.5:
        short_doc = result["document"][:50] + "..." if len(result["document"]) > 50 else result["document"]
        results.append({
            "документ": short_doc,
            "словосочетание": query,
            "позиция": result["positions"],
            "косинусное_расстояние": result["distance"]
        })


# Выводим результаты
print("\nДОКУМЕНТ".ljust(50), "СЛОВОСОЧЕТАНИЕ".ljust(20), "ВЫВОД")
for res in results:
    print(
        res["документ"].ljust(50),
        res["словосочетание"].ljust(20),
        f"позиция: {res['позиция']}".ljust(25),
        f"косинусное расстояние: {res['косинусное_расстояние']:.3f}"
    )

out_dir = 'model'

# Сохраняем результаты
os.makedirs(out_dir, exist_ok=True)
joblib.dump(results, f'{out_dir}/search_results.pkl')


# Сохраняем модель в папку ST_model/
os.makedirs(out_dir, exist_ok=True)  # Создаём папку, если её нет
ST_model.save(f'{out_dir}/semantic_search_model')
print(f"Модель сохранена в {out_dir}/semantic_search_model")



# Проверяем, существует ли сохранённая модель
if os.path.exists(f'{out_dir}/semantic_search_model'):
    ST_model = SentenceTransformer(f'{out_dir}/semantic_search_model')
    print("Модель загружена из локального файла")
else:
    ST_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Модель загружена из интернета")
    os.makedirs(out_dir, exist_ok=True)
    ST_model.save(f'{out_dir}/semantic_search_model')
    print(f"Модель сохранена в {out_dir}/semantic_search_model")