import pickle

from flask import Flask, render_template, request, jsonify, redirect, url_for

# Создаём Flask приложение
app = Flask(__name__)

# Путь к файлам модели и пайплайна
MODEL_FILE = '../ml/model/model.pkl'
PIPELINE_FILE = '../ml/model/pipeline.pkl'

# === Загружаем модель и пайплайн только один раз ===
model = None
pipeline = None


def load_model_and_pipeline():
    global model, pipeline
    if model is None or pipeline is None:
        # Загрузка сохранённых модели и пайплайна
        with open(MODEL_FILE, 'rb') as model_file:
            model = pickle.load(model_file)

        with open(PIPELINE_FILE, 'rb') as pipeline_file:
            pipeline = pickle.load(pipeline_file)
    return model, pipeline


# === GET /index ===
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


# === POST /predict ===
@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из формы
    features = request.form.getlist('features')

    # Преобразуем введённые данные в числовой формат
    try:
        features = list(map(float, features))
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter numeric values.'}), 400
    try:

        # Загружаем модель и пайплайн (один раз)
        model, pipeline = load_model_and_pipeline()

        # Применяем пайплайн для предобработки данных (масштабирование, PCA и т.д.)
        processed_features = pipeline.transform([features])

        # Получаем предсказание от модели
        prediction = model.predict(processed_features)

        # Редирект на /index с результатом предсказания
        return redirect(url_for('index', prediction=prediction[0]))
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html')

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True, port=8081)
