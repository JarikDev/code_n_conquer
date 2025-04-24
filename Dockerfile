FROM python:3.10-slim

# Рабочая директория
WORKDIR /app

# Копируем зависимости и исходники
COPY requirements.txt ./
COPY src ./src
COPY src/ml ./src/ml

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Переменные окружения для запуска Flask
ENV FLASK_APP=cmd/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Открываем порт
EXPOSE 8081

# Запуск приложения
CMD ["flask", "run"]
