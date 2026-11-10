from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ==================================================
# 1. Настройки проекта
# ==================================================

DATA_PATH = Path("data/wine.csv")
REPORTS_DIR = Path("reports")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BEST_K = 3


# ==================================================
# 2. Загрузка данных
# ==================================================

print("\n" + "=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Файл с данными не найден: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Размер таблицы:", df.shape)
print("\nПервые пять строк:")
print(df.head())

print("\nРаспределение исходных классов:")
print(df["target"].value_counts().sort_index())

print("\nКоличество пропущенных значений:", df.isnull().sum().sum())


# ==================================================
# 3. Типы данных и статистика
# ==================================================

print("\n" + "=" * 60)
print("2. ТИПЫ ДАННЫХ И СТАТИСТИКА")
print("=" * 60)

print("\nТипы данных:")
print(df.dtypes)

print("\nСтатистические характеристики:")
print(df.describe().round(2).T)


# ==================================================
# 4. Распределение признаков
# ==================================================

print("\n" + "=" * 60)
print("3. РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ")
print("=" * 60)

X = df.drop(columns=["target"])

X.hist(
    figsize=(15, 12),
    bins=20,
    edgecolor="black"
)

plt.suptitle(
    "Распределение химических признаков Wine",
    fontsize=16
)
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "feature_distributions.png",
    dpi=150
)
plt.close()

print("График сохранён: reports/feature_distributions.png")


# ==================================================
# 5. Корреляционная матрица
# ==================================================

print("\n" + "=" * 60)
print("4. КОРРЕЛЯЦИОННАЯ МАТРИЦА")
print("=" * 60)

plt.figure(figsize=(13, 10))

sns.heatmap(
    X.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Корреляционная матрица признаков Wine")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "correlation_matrix.png",
    dpi=150
)
plt.close()

print("График сохранён: reports/correlation_matrix.png")


# ==================================================
# 6. Стандартизация признаков
# ==================================================

print("\n" + "=" * 60)
print("5. СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Размер исходных признаков:", X.shape)
print("Размер после стандартизации:", X_scaled.shape)

print("\nСредние значения:")
print(X_scaled.mean(axis=0).round(3))

print("\nСтандартные отклонения:")
print(X_scaled.std(axis=0).round(3))


# ==================================================
# 7. Подбор количества кластеров
# ==================================================

print("\n" + "=" * 60)
print("6. ПОДБОР КОЛИЧЕСТВА КЛАСТЕРОВ")
print("=" * 60)

results = []

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "silhouette": score,
        "inertia": model.inertia_
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

best_result = results_df.loc[
    results_df["silhouette"].idxmax()
]

print("\nЛучшее количество кластеров:")
print(best_result)

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["k"],
    results_df["silhouette"],
    marker="o"
)

plt.xticks(range(2, 8))
plt.xlabel("Количество кластеров (k)")
plt.ylabel("Silhouette Score")
plt.title("Подбор количества кластеров для KMeans")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "silhouette_scores.png",
    dpi=150
)
plt.close()

print("График сохранён: reports/silhouette_scores.png")


# ==================================================
# 8. Обучение итоговой модели
# ==================================================

print("\n" + "=" * 60)
print("7. ИТОГОВАЯ КЛАСТЕРИЗАЦИЯ")
print("=" * 60)

model = KMeans(
    n_clusters=BEST_K,
    random_state=RANDOM_STATE,
    n_init=10
)

clusters = model.fit_predict(X_scaled)

df_clustered = df.copy()
df_clustered["cluster"] = clusters

score = silhouette_score(X_scaled, clusters)

print("Количество кластеров:", BEST_K)
print("Silhouette Score:", round(score, 4))

print("\nРазмеры кластеров:")
print(df_clustered["cluster"].value_counts().sort_index())

print("\nСравнение кластеров с исходными классами:")
print(pd.crosstab(
    df_clustered["cluster"],
    df_clustered["target"]
))

df_clustered.to_csv(
    "data/wine_clustered.csv",
    index=False
)

print("\nРезультаты сохранены: data/wine_clustered.csv")


# ==================================================
# 9. Визуализация кластеров с помощью PCA
# ==================================================

print("\n" + "=" * 60)
print("8. ВИЗУАЛИЗАЦИЯ PCA")
print("=" * 60)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50,
    alpha=0.8
)

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("Визуализация кластеров Wine с помощью PCA")
plt.colorbar(label="Номер кластера")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "clusters_pca.png",
    dpi=150
)
plt.close()

print("Объяснённая дисперсия:")
print(pca.explained_variance_ratio_.round(4))

print(
    "Суммарная:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

print("График сохранён: reports/clusters_pca.png")


# ==================================================
# 10. Анализ профилей кластеров
# ==================================================

print("\n" + "=" * 60)
print("9. ПРОФИЛИ КЛАСТЕРОВ")
print("=" * 60)

features = df.drop(columns=["target"]).columns

profiles = (
    df_clustered
    .groupby("cluster")[features]
    .mean()
)

print("\nСредние значения признаков:")
print(profiles.round(2).T)

profiles.round(3).to_csv(
    REPORTS_DIR / "cluster_profiles.csv"
)

print("\nПрофили сохранены: reports/cluster_profiles.csv")


# ==================================================
# 11. Итоговая информация
# ==================================================

print("\n" + "=" * 60)
print("10. РАБОТА ЗАВЕРШЕНА")
print("=" * 60)

print("Использовано объектов:", len(df))
print("Количество признаков:", X.shape[1])
print("Количество кластеров:", BEST_K)
print("Silhouette Score:", round(score, 4))

print("\nСозданные файлы:")

for path in sorted(REPORTS_DIR.iterdir()):
    if path.is_file():
        print(f"  {path}")

print("\nИтоговая таблица: data/wine_clustered.csv")
