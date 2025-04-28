import joblib
import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from src.ml.semantic_analysis import semantic_search_in_document

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
base_path = "../ml/model/"
search_results = joblib.load(f"{base_path}search_results.pkl")

test_data: DataFrame = pd.read_csv("data/search_examples.csv")

test_data['Вывод'] = test_data['Вывод'].fillna("Не найдено")
test_data['Вывод'] = test_data['Вывод'].apply(lambda x: x.replace("\n", " "))

print(test_data.head())
print(test_data.info())

# Загрузка модели sentence-transformers
model = SentenceTransformer('intfloat/multilingual-e5-large')


def get_result(text, phrase, model):
    print(f'text: {text}, phrase: {phrase}')
    result = semantic_search_in_document(text, phrase, model)
    print(result)
    return f"Позиция: {result['positions']} Вероятность: {round(result['score'], 3)}"


test_data["Результат"] = test_data.apply(lambda x: get_result(x[0], x[1], model), axis=1)
test_data["Итог"] = test_data.apply(lambda x: "Правильно" if x[2] == x[3] else "Ошибка", axis=1)

print(test_data.head())
print(test_data.info())
test_data.to_csv("data/test_out.csv")
