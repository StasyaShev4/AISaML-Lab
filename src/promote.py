import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient

p = argparse.ArgumentParser()
p.add_argument('--run_id', required=True, help='Run ID модели для регистрации')
p.add_argument('--model_name', default='wine-clusterer', help='Имя модели в реестре')
p.add_argument('--alias', default='champion', help='Alias для версии (например, champion)')
a = p.parse_args()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
client = MlflowClient()

# Формируем URI модели из run_id (имя артефакта 'model' как в стандартном train.py)
model_uri = f"runs:/{a.run_id}/model"

print(f"Регистрируем модель из run {a.run_id} как '{a.model_name}'...")
# register_model создает новую версию, если модель с таким именем уже существует
mv = mlflow.register_model(model_uri, a.model_name)
print(f"Зарегистрирована версия: {mv.version}")

print(f"Назначаем alias '{a.alias}' для версии {mv.version}...")
client.set_registered_model_alias(a.model_name, a.alias, mv.version)
print(f"Успешно: {a.model_name}@{a.alias} теперь указывает на версию {mv.version}")
