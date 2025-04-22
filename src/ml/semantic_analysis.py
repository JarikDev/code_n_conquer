import pandas as pd
from pandas import DataFrame

# TODO здесь сделать модель обучение и запиклить например пайплайн и модель
def load_data(url):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(url)

# Загрузка данных
data: DataFrame = load_data("https://drive.google.com/file/d/1rmMsqM5Fps4XnnfWPES6JPNNUSDcdr1H/view?usp=sharing")

print('Посмотрим на данные')
print(data.head())
print(data.info())
print('Количество пропусков в колонках')
print(data.isna().sum())