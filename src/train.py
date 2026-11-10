import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine_Clustering_v30")


df = pd.read_csv("data/wine.csv")

X = df.drop(columns=["target"])

# Шесть конфигураций для k от 2 до 7
configs = [{"n_clusters": k, "random_state": 42, "n_init": 10} for k in range(2, 8)]

for config in configs:
    with mlflow.start_run(run_name=f"KMeans_k{config['n_clusters']}"):
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(**config))
        ])
        
        pipeline.fit(X)
        
        labels = pipeline.named_steps["kmeans"].labels_
        sil_score = silhouette_score(X, labels)
        inertia = pipeline.named_steps["kmeans"].inertia_
        
        print(f"k={config['n_clusters']}: silhouette={sil_score:.6f}, inertia={inertia:.6f}")
        
        mlflow.log_params(config)
        mlflow.log_metrics({"silhouette_score": sil_score, "inertia": inertia})
        
        mlflow.sklearn.log_model(pipeline, "model")
        print(f"Модель k={config['n_clusters']} успешно залогирована.\n")
