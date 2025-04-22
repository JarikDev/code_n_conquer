import pickle

from flask import Flask, request, jsonify

# Создаём Flask приложение
app = Flask(__name__)

# Путь к файлам модели и пайплайна
MODEL_FILE = '../ml/model/model.pkl'
PIPELINE_FILE = '../ml/model/pipeline.pkl'


# === Загрузка модели и пайплайна ===
def load_model_and_pipeline():
    # Загрузка сохранённых модели и пайплайна
    with open(MODEL_FILE, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(PIPELINE_FILE, 'rb') as pipeline_file:
        pipeline = pickle.load(pipeline_file)

    return model, pipeline


# === GET /status ===
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ok', 'message': 'Code & Conquer BI4ezzzz...'}), 200


# === POST /predict ===
@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    data = request.get_json()

    # Проверяем, что данные присутствуют
    if not data or 'features' not in data:
        return jsonify({'error': 'No data provided or "features" field is missing'}), 400

    features = data['features']

    # Загружаем модель и пайплайн
    model, pipeline = load_model_and_pipeline()

    # Применяем пайплайн для предобработки данных (масштабирование, PCA, и т.д.)
    processed_features = pipeline.transform([features])

    # Получаем предсказание от модели
    prediction = model.predict(processed_features)

    # Возвращаем результат предсказания
    return jsonify({'prediction': prediction.tolist()}), 200


# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True, port=8081)
