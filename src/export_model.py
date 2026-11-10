import json
import os
import shutil
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
c = MlflowClient()

model_name = 'wine-clusterer'
v = c.get_model_version_by_alias(model_name, 'champion')

print(f"Экспорт модели {model_name} версии {v.version} (run: {v.run_id})...")

model = mlflow.sklearn.load_model(f'models:/{model_name}/{v.version}')
out = Path('exported_model')
if out.exists():
    shutil.rmtree(out)
    
mlflow.sklearn.save_model(
    model, 
    str(out), 
    pip_requirements=Path('requirements.txt').read_text().splitlines()
)


df = pd.read_csv("data/wine.csv")
X = df.drop(columns=["target"])

metadata = {
    'name': model_name,
    'version': v.version,
    'run_id': v.run_id,
    'features': X.columns.tolist(),
    'task': 'clustering'
}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))


sample = X.iloc[:3]
request = {'instances': sample.to_dict(orient='records')}
Path('sample_request.json').write_text(json.dumps(request, indent=2))


expected_preds = model.predict(sample).tolist()
Path('expected_predictions.json').write_text(json.dumps(expected_preds))

print("Модель успешно экспортирована в папку exported_model/")
print(json.dumps(metadata, indent=2))
