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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Загрузка данных
categories = ['sci.space', 'rec.sport.hockey']
data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# Векторизация
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(data.data)
X_test = vectorizer.transform(test_data.data)

# Модель
model = LogisticRegression(max_iter=1000)
model.fit(X_train, data.target)

# Метрики
y_pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(test_data.target, y_pred),
    "report": classification_report(test_data.target, y_pred, output_dict=True)
}

# Сохранение модели и метрик
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(metrics, "model/metrics.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")