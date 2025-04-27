# Используем официальный образ Python
FROM python:3.8-slim

# Устанавливаем зависимости
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Копируем код в контейнер
COPY . /app

# Запускаем Flask приложение
CMD ["python", "app.py"]
