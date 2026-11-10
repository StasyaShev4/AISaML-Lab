# Лабораторная работа №4: MLOps. Учет экспериментов и контейнеризация

**Вариант 30:** Кластеризация образцов вина

## Описание задачи


### Датасет
- **Источник**: UCI Wine Recognition Data (https://archive.ics.uci.edu/dataset/109/wine)
- **Объектов**: 178
- **Признаков**: 13 химических характеристик (alcohol, malic_acid, ash, и др.)
- **Задача**: Кластеризация (обучение без учителя, target не используется при обучении)
- **Метрика**: Silhouette Score

#### Технологии
  * Python 3.12
  * MLflow 3.18.0
  * scikit-learn 1.9.1
  * FastAPI 0.162.2
  * Docker 29.1.3
  * PostgreSQL 16 (для MLflow backend)


### Быстрый старт

#### 1. Установка зависимостей
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Запуск MLflow с PostgreSQL

```bash
docker compose up -d postgres
mlflow server \
  --backend-store-uri postgresql+psycopg2://mlflow:mlflow_password@localhost:5432/mlflow \
  --artifacts-destination ./mlflow-data \
  --host 127.0.0.1 \
  --port 5000
```

#### 3. Обучение моделей (k=2...7)

```bash
python src/train.py
```
Скрипт создает 6 запусков в MLflow с метриками silhouette_score и inertia.

#### 4. Регистрация моделей

```bash
# Регистрация версии 1 (previous, k=4)
python src/promote.py --run_id <run_id_k4> --model_name wine-clusterer --alias previous

# Регистрация версии 2 (champion, k=2 - лучшая по silhouette)
python src/promote.py --run_id <run_id_k2> --model_name wine-clusterer --alias champion
```

#### 5. Экспорт и запуск API

```bash
python src/export_model.py
sudo docker build -t wine-api:v1 .
sudo docker run -d --name wine-api -p 127.0.0.1:8000:8000 wine-api:v1
```

#### 6. Проверка API

```bash
# Health check
curl http://127.0.0.1:8000/health

# Метаданные модели
curl http://127.0.0.1:8000/metadata

# Предсказание кластеров
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @sample_request.json

# Автоматическая проверка
python src/check_api.py
```
#### 7. Запуск через Docker Compose

```bash
sudo docker compose up -d --build
sudo docker compose ps  # Проверить статус healthy
curl http://127.0.0.1:8000/health
sudo docker compose down
```