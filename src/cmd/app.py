# Здась будет код прилождения, бакенд

from flask import Flask, request, jsonify
import numpy as np
import json

# Создаём Flask приложение
app = Flask(__name__)

# === GET /status ===
# Этот эндпоинт проверяет состояние приложения.
@app.route('/status', methods=['GET'])
def status():
    # Возвращаем сообщение о том, что приложение работает
    return jsonify({'status': 'ok', 'message': 'The application is running.'}), 200


# === POST /predict ===
# Этот эндпоинт будет принимать входные данные через POST-запрос,
# например, данные для предсказания с моделью, и возвращать результат.
@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    data = request.get_json()

    # Проверяем, что данные присутствуют
    if not data or 'features' not in data:
        return jsonify({'error': 'No data provided or "features" field is missing'}), 400

    # Пример обработки входных данных:
    # Для простоты мы возьмём список признаков, переданных в "features",
    # и просто посчитаем их сумму как "предсказание".
    features = data['features']

    # Пример "предсказания" (вычисляем сумму входных данных)
    prediction = np.sum(features)

    # Возвращаем результат в формате JSON
    return jsonify({'prediction': prediction}), 200


# Запуск приложения на порту 5000
if __name__ == '__main__':
    app.run(debug=True, port=8081)
