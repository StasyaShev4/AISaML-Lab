import json
import urllib.request
from pathlib import Path

# Загружаем тестовый запрос и ожидаемые предсказания
body = Path('sample_request.json').read_bytes()
expected = json.loads(Path('expected_predictions.json').read_text())

# Отправляем запрос к API
req = urllib.request.Request(
    'http://127.0.0.1:8000/predict',
    data=body,
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(req, timeout=30) as r:
    result = json.load(r)

# Для кластеризации сравниваем cluster_ids
print("Результат API:", result)
print("Ожидаемые кластеры:", expected)

if result['cluster_ids'] == expected:
    print('PASS: API возвращает те же кластеры, что и экспортированная модель')
else:
    print('FAIL: Несоответствие предсказаний')
    print(f"  API: {result['cluster_ids']}")
    print(f"  Ожидалось: {expected}")
