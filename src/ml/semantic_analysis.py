import pandas as pd
from pandas import DataFrame
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
# TODO здесь сделать модель обучение и запиклить например пайплайн и модель
# def load_data(url):
#     url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
#     return pd.read_csv(url)
#
# # Загрузка данных
# data: DataFrame = load_data("https://drive.google.com/file/d/1rmMsqM5Fps4XnnfWPES6JPNNUSDcdr1H/view?usp=sharing")
#
# print('Посмотрим на данные')
# print(data.head())
# print(data.info())
# print('Количество пропусков в колонках')
# print(data.isna().sum())

# файлы оказались не толстыми, запихнул в проект

data_files = ['./data/1.csv','./data/2.csv','./data/3.csv','./data/4.csv','./data/5.csv','./data/6.csv']

def get_basic_info_from_file(path):
    data: DataFrame = pd.read_csv(path)
    print('Посмотрим на данные')
    print(data.head())
    print(data.info())
    print('Количество пропусков в колонках')
    print(data.isna().sum())

for file in data_files:
    get_basic_info_from_file(file)

# Предположим, что у вас есть готовый пайплайн и модель
scaler = StandardScaler()
pca = PCA(n_components=2)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Создаём пайплайн
pipeline = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('classifier', clf)
])

# Обучение модели на произвольных данных
# Здесь вам нужно использовать ваши реальные данные для обучения
X_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_train = [0, 1, 0]
pipeline.fit(X_train, y_train)

# Сохраняем модель и пайплайн в pickle файл
with open('./model/model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)  # Сохраняем модель

with open('./model/pipeline.pkl', 'wb') as pipeline_file:
    pickle.dump(pipeline, pipeline_file)  # Сохраняем пайплайн