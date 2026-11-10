import json
import logging
import time
from pathlib import Path
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Wine Clusterer API', version='1.0')
model = mlflow.sklearn.load_model('/app/exported_model')
meta = json.loads(Path('/app/exported_model/metadata.json').read_text())
log = logging.getLogger('uvicorn.error')

class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1, max_length=100)

@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': meta['version']}

@app.get('/metadata')
def metadata():
    return meta

@app.post('/predict')
def predict(body: PredictRequest):
    start = time.perf_counter()
    required = set(meta['features'])
    
    
    for i, row in enumerate(body.instances):
        if set(row.keys()) != required:
            raise HTTPException(422, detail={
                'message': f'Row {i} must contain exactly these features',
                'features': meta['features'],
                'provided': list(row.keys())
            })
            
    X = pd.DataFrame(body.instances)[meta['features']]
    
    if not np.isfinite(X.to_numpy()).all():
        raise HTTPException(422, detail='All feature values must be finite')
        
    
    result = model.predict(X).astype(int).tolist()
    elapsed = (time.perf_counter() - start) * 1000
    
    log.info('prediction model_version=%s rows=%s latency_ms=%.3f', meta['version'], len(X), elapsed)
    
    return {
        'cluster_ids': result,
        'model_version': meta['version'],
        'latency_ms': round(elapsed, 3)
    }
