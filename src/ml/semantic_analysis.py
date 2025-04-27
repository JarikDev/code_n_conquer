# Загружаем библиотеки
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import joblib
import os



# Ссылка на raw-версию CSV
# Список файлов CSV для слияния
csv_files = [f"data/{i}.csv" for i in range(1, 7)]

# Чтение и слияние CSV файлов в один DataFrame
dfs = [pd.read_csv(file) for file in csv_files]

# Слияние всех DataFrame в один
merged_df = pd.concat(dfs, ignore_index=True)

# Копия датасета
text_df = merged_df.copy()



# Выводим информацию о данных
print('Данные имеют следующую размерность: ')
print(f'Количество строк: {text_df.shape[0]}; [Исключённый текст] признаков (столбцов): {text_df.shape[1]}.')
print(' - ' * 40)

# Выводим первые 10 строк
print('\nВыводим первые 10 строк')
print(text_df.head(10))
print('0.0s')



# Загружаем модель sentence-transformers
model = SentenceTransformer('intfloat/multilingual-e5-large')
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

    # Очищаем документ и запрос
    doc_cleaned = clean_text(document)
    query_cleaned = clean_text(query)

    # Проверяем, что запрос содержит не более 2 слов
    if len(query_cleaned.split()) > 2:
        return {"error": "Запрос должен содержать не более 2 слов."}


    # Разделяем документ на слова
    doc_words = doc_cleaned.split()

    if not doc_words:
        return {
            'document': document, 
            'score': 0.0, 
            'positions': 'Слова Не Найдены В ТЕКСТЕ', 
            'matched_word': ''
            }

    # Создаём эмбеддинги для запроса и каждого слова в документе
    query_embedding = model.encode(query_cleaned, convert_to_tensor = True)
    word_embeddings = model.encode(doc_words, convert_to_tensor = True)

    # Вычисляем косинусное сходство между запросом и каждым словом
    cosine_scores = util.cos_sim(query_embedding, word_embeddings)[0]


    # Проверяем, есть ли точное совпадение (например, "дочь" в "доченька")
    best_match_idx = None
    best_score = -1

    for i, word in enumerate(doc_words):
        score = cosine_scores[i].item()

        # Даём приоритет словам, содержащим запрос как подстроку
        if query_cleaned in word:
            score += 1.0  # Добавляем бонус, чтобы выбрать это слово

        if score > best_score:
            best_score = score
            best_match_idx = i

    if best_match_idx is None or best_score < 0.05:
        return {
            'document': document, 
            'score': 0.0,
            'positions': 'Слова Не Найдены В ТЕКСТЕ', 
            'matched_word': ''
            }

    best_word = doc_words[best_match_idx]

    # Устанавливаем вероятность 1.0, если сходство выше порога
    display_score = 1.0 if best_score >= 0.5 else best_score

    # Вычисляем позиции в символах в исходной строке
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

    return {
        'document': document,
        'score': display_score,
        'positions': positions_str,
        'matched_word': matched_word
    }





# Заданные списки строк (документы) и словосочетаний
documents = [
    "доченька твоя совсем большая стала",
    "в лесу растёт высокое дерево",
    "санк петербург красивый город",
    "машина едет по дороге быстро",
    "новый автобиль стоит в гараже",
    "учитель объясняет урок детям",
    "обучение проходит в школе ежегодно",
    "позвони мне завтра утром",
    "в болнице работает мой друг",
    "телефон лежит на столе рядом",
    "россия большая и красивая страна",
    "портфель лежит в шкафу дома",
    "институт находится в центре города"
]


queries = [
    "дочь",
    "дерево",
    "петербург",
    "машина",
    "автобиль",
    "учитель",
    "обучение",
    "звонить",
    "болница",
    "телефон",
    "россия",
    "портфель",
    "институт"
]



# Проверяем, что длины списков совпадают
assert len(documents) == len(queries), "Длина списков documents и queries должна совпадать!"



# Выполняем поиск для каждой пары (document, query)
results = []

for doc, query in zip(documents, queries):

    # Применяем функцию с моделью
    result = semantic_search_in_document(doc, query, model)

    # Укорачиваем документ до 50 символов для компактного отображения
    short_doc = result["document"][:50] + "..." if len(result["document"]) > 50 else result["document"]
    results.append({
        "документ": short_doc,
        "словосочетание": query,
        "позиция": result["positions"],
        "вероятность": result["score"]
    })



# Выводим результаты в формате таблицы
print("\nДОКУМЕНТ".ljust(50), "СЛОВОСОЧЕТАНИЕ".ljust(20), "ВЫВОД")
for res in results:

    print(
        res["документ"].ljust(50),
        res["словосочетание"].ljust(20),
        f"позиция: {res['позиция']}".ljust(25),
        f"вероятность: {res['вероятность']:.1f}"
    )




# Сохраняем результаты
os.makedirs('model', exist_ok=True)
joblib.dump(results, 'model/search_results.pkl')