from IPython.display import display
# --- Кодовая ячейка исходного ноутбука: 4 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import tempfile

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, Binarizer, LabelEncoder, label_binarize
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


# --- Кодовая ячейка исходного ноутбука: 6 ---
DATA_PATH = "Dry_Bean_Dataset.xlsx"
df = pd.read_excel(DATA_PATH)

print("Размер датасета:", df.shape)

print("\nТипы данных:")
print(df.dtypes)

print("\nКоличество пропусков в каждом столбце:")
print(df.isnull().sum())

print("\nОбщее количество пропущенных значений:")
print(df.isnull().sum().sum())

print("\nКоличество полностью дублирующихся строк:")
print(df.duplicated().sum())

print("\nКоличество бесконечных значений:")
print(
    df.select_dtypes(include="number")
      .isin([float("inf"), float("-inf")])
      .sum()
      .sum()
)

print("\nОсновные статистические характеристики:")
print(df.describe().round(2))


# --- Кодовая ячейка исходного ноутбука: 8 ---
duplicate_rows = df[df.duplicated(keep=False)]

print("Количество строк, участвующих в дублировании:",
      len(duplicate_rows))

print("\nРаспределение классов среди дублирующихся строк:")
print(duplicate_rows["Class"].value_counts())

print("\nКоличество различных классов у одинаковых наборов признаков:")

feature_cols = df.columns.drop("Class")

class_counts = (
    duplicate_rows
    .groupby(list(feature_cols), dropna=False)["Class"]
    .nunique()
)

print("Одинаковые признаки с разными классами:",
      (class_counts > 1).sum())


# --- Кодовая ячейка исходного ноутбука: 10 ---
rows_before = len(df)

df = df.drop_duplicates().copy()

rows_after = len(df)

print("Количество строк до удаления:", rows_before)
print("Удалено дублирующихся строк:", rows_before - rows_after)
print("Количество строк после удаления:", rows_after)
print("Оставшиеся полные дубликаты:", df.duplicated().sum())


# --- Кодовая ячейка исходного ноутбука: 13 ---
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Размер исходного датасета:", df.shape)
print("Обучающая выборка X_train:", X_train.shape)
print("Тестовая выборка X_test:", X_test.shape)
print("Количество меток в обучающей выборке:", y_train.shape[0])
print("Количество меток в тестовой выборке:", y_test.shape[0])


# --- Кодовая ячейка исходного ноутбука: 15 ---
class_distribution = pd.DataFrame({
    "Весь датасет, %": y.value_counts(normalize=True) * 100,
    "Обучающая выборка, %": y_train.value_counts(normalize=True) * 100,
    "Тестовая выборка, %": y_test.value_counts(normalize=True) * 100
}).sort_index()

print(class_distribution.round(2))


# --- Кодовая ячейка исходного ноутбука: 18 ---
def evaluate_model(model, X_test, y_test, model_name, verbose=False):
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "Модель": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1-score": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )
    }

    if verbose:
        print(f"\n{model_name}")
        for metric, value in metrics.items():
            if metric != "Модель":
                print(f"{metric}: {value:.4f}")

    return metrics


# --- Кодовая ячейка исходного ноутбука: 20 ---
nb_models = {
    "GaussianNB": make_pipeline(
        StandardScaler(),
        GaussianNB()
    ),

    "MultinomialNB": make_pipeline(
        MinMaxScaler(),
        MultinomialNB()
    ),

    "ComplementNB": make_pipeline(
        MinMaxScaler(),
        ComplementNB()
    ),

    "BernoulliNB": make_pipeline(
        MinMaxScaler(),
        Binarizer(threshold=0.5),
        BernoulliNB()
    )
}

nb_results = []

for name, model in nb_models.items():
    model.fit(X_train, y_train)
    metrics = evaluate_model(
        model, X_test, y_test, name,
        verbose=True
    )
    nb_results.append(metrics)


# --- Кодовая ячейка исходного ноутбука: 23 ---
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

tree_results = evaluate_model(
    decision_tree, X_test, y_test, "Decision Tree"
)


# --- Кодовая ячейка исходного ноутбука: 25 ---
lda_model = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis()
)

lda_model.fit(X_train, y_train)

lda_results = evaluate_model(
    lda_model, X_test, y_test, "Linear Discriminant Analysis (LDA)"
)


# --- Кодовая ячейка исходного ноутбука: 27 ---
svm_model = CalibratedClassifierCV(
    estimator=make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", random_state=42)
    ),
    method="sigmoid",
    cv=5
)

svm_model.fit(X_train, y_train)

svm_results = evaluate_model(
    svm_model,
    X_test,
    y_test,
    "Support Vector Machine (SVM)"
)


# --- Кодовая ячейка исходного ноутбука: 29 ---
knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn_model.fit(X_train, y_train)

knn_results = evaluate_model(
    knn_model, X_test, y_test, "k-Nearest Neighbors (k-NN)"
)


# --- Кодовая ячейка исходного ноутбука: 30 ---
metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

all_results_df = pd.DataFrame([
    *nb_results,
    tree_results,
    lda_results,
    svm_results,
    knn_results
])

all_results_df[metric_columns] = all_results_df[metric_columns].round(4)

display(all_results_df)


# --- Кодовая ячейка исходного ноутбука: 33 ---
def tune_model(name, estimator, param_grid, X_train, y_train,
               X_test, y_test, scoring="f1_weighted", cv=5,
               n_jobs=-1, refit=True, verbose=False,
):
   
    print(f"Настройка {name}...")

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
    )
    search.fit(X_train, y_train)

    print(f"Лучшие параметры: {search.best_params_}")
    print(f"Лучший CV F1-score: {search.best_score_:.4f}\n")

    tuning_info = {
        "Модель": name,
        "Лучшие параметры": str(search.best_params_),
        "CV F1-score": search.best_score_,
    }

    test_metrics = evaluate_model(
        search.best_estimator_,
        X_test, y_test,
        f"{name} (настроенная)",
        verbose=verbose,
    )

    return search.best_estimator_, tuning_info, test_metrics


# --- Кодовая ячейка исходного ноутбука: 34 ---
nb_searches = {
    "GaussianNB": {
        "model": make_pipeline(StandardScaler(), GaussianNB()),
        "params": {"gaussiannb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    },
    "MultinomialNB": {
        "model": make_pipeline(MinMaxScaler(), MultinomialNB()),
        "params": {
            "multinomialnb__alpha": [0.1, 0.5, 1.0, 2.0],
            "multinomialnb__fit_prior": [True, False],
        },
    },
    "ComplementNB": {
        "model": make_pipeline(MinMaxScaler(), ComplementNB()),
        "params": {
            "complementnb__alpha": [0.1, 0.5, 1.0, 2.0],
            "complementnb__fit_prior": [True, False],
            "complementnb__norm": [True, False],
        },
    },
    "BernoulliNB": {
        "model": make_pipeline(StandardScaler(), Binarizer(threshold=0.0), BernoulliNB()),
        "params": {
            "bernoullinb__alpha": [0.1, 0.5, 1.0, 2.0],
            "bernoullinb__fit_prior": [True, False],
            "binarizer__threshold": [0.25, 0.5, 0.75],
        },
    },
}

best_nb_models = {}
nb_tuning_results = []
nb_test_results = []

for name, config in nb_searches.items():
    best_model, tuning_info, test_metrics = tune_model(
        name=name,
        estimator=config["model"],
        param_grid=config["params"],
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    best_nb_models[name] = best_model
    nb_tuning_results.append(tuning_info)
    nb_test_results.append(test_metrics)


nb_test_df = pd.DataFrame(nb_test_results)
metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
nb_test_df[metric_columns] = nb_test_df[metric_columns].round(4)
display(nb_test_df)


# --- Кодовая ячейка исходного ноутбука: 36 ---
best_models = {}
tuning_results = []
test_results = []

best_dt, dt_tune, dt_test = tune_model(
    "Decision Tree",
    DecisionTreeClassifier(random_state=42),
    {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    },
    X_train, y_train, X_test, y_test,
)
best_models["Decision Tree"] = best_dt
tuning_results.append(dt_tune)
test_results.append(dt_test)


best_lda, lda_tune, lda_test = tune_model(
    "LDA",
    make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    [
        {"lineardiscriminantanalysis__solver": ["svd"]},
        {
            "lineardiscriminantanalysis__solver": ["lsqr", "eigen"],
            "lineardiscriminantanalysis__shrinkage": ["auto", 0.1, 0.5, 0.9],
        },
    ],
    X_train, y_train, X_test, y_test,
)
best_models["LDA"] = best_lda
tuning_results.append(lda_tune)
test_results.append(lda_test)


best_svm, svm_tune, svm_test = tune_model(
    "SVM",
    CalibratedClassifierCV(
        estimator=make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
        method="sigmoid",
        cv=3,
    ),
    {
        "estimator__svc__C": [0.1, 1, 10],
        "estimator__svc__gamma": ["scale", 0.01, 0.1],
    },
    X_train, y_train, X_test, y_test,
    cv=3,
)
best_models["SVM"] = best_svm
tuning_results.append(svm_tune)
test_results.append(svm_test)


best_knn, knn_tune, knn_test = tune_model(
    "k-NN",
    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    {
        "kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11],
        "kneighborsclassifier__weights": ["uniform", "distance"],
        "kneighborsclassifier__p": [1, 2],
    },
    X_train, y_train, X_test, y_test,
)
best_models["k-NN"] = best_knn
tuning_results.append(knn_tune)
test_results.append(knn_test)


test_df = pd.DataFrame(test_results)
test_df[metric_columns] = test_df[metric_columns].round(4)
print("Метрики настроенных моделей на тестовой выборке:")
display(test_df)


# --- Кодовая ячейка исходного ноутбука: 38 ---
baseline_results = [
    {
        "Модель": "GaussianNB",
        "Accuracy": 0.8970,
        "Precision": 0.8997,
        "Recall": 0.8970,
        "F1-score": 0.8972,
        "ROC-AUC": 0.9899
    },
    {
        "Модель": "MultinomialNB",
        "Accuracy": 0.5150,
        "Precision": 0.5638,
        "Recall": 0.5150,
        "F1-score": 0.4467,
        "ROC-AUC": 0.9589
    },
    {
        "Модель": "ComplementNB",
        "Accuracy": 0.5951,
        "Precision": 0.4756,
        "Recall": 0.5951,
        "F1-score": 0.4954,
        "ROC-AUC": 0.8761
    },
    {
        "Модель": "BernoulliNB",
        "Accuracy": 0.6884,
        "Precision": 0.7055,
        "Recall": 0.6884,
        "F1-score": 0.6827,
        "ROC-AUC": 0.9328
    },
    {
        "Модель": "Decision Tree",
        "Accuracy": 0.8955,
        "Precision": 0.8954,
        "Recall": 0.8955,
        "F1-score": 0.8953,
        "ROC-AUC": 0.9357
    },
    {
        "Модель": "LDA",
        "Accuracy": 0.9033,
        "Precision": 0.9130,
        "Recall": 0.9033,
        "F1-score": 0.9046,
        "ROC-AUC": 0.9912
    },
    {
        "Модель": "SVM",
        "Accuracy": 0.9210,
        "Precision": 0.9214,
        "Recall": 0.9210,
        "F1-score": 0.9211,
        "ROC-AUC": 0.9933
    },
    {
        "Модель": "k-NN",
        "Accuracy": 0.9155,
        "Precision": 0.9163,
        "Recall": 0.9155,
        "F1-score": 0.9157,
        "ROC-AUC": 0.9811
    }
]

baseline_df = pd.DataFrame(baseline_results)

tuned_nb = nb_test_df.copy()
tuned_other = test_df.copy()


tuned_nb["Модель"] = tuned_nb["Модель"].str.replace(
    " (настроенная)", "", regex=False
)
tuned_other["Модель"] = tuned_other["Модель"].str.replace(
    " (настроенная)", "", regex=False
)

tuned_df = pd.concat(
    [tuned_nb, tuned_other],
    ignore_index=True
)

metric_columns = [
    "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"
]


baseline_df = baseline_df.set_index("Модель")

tuned_df = tuned_df.set_index("Модель")


comparison_df = pd.concat(
    [
        baseline_df[metric_columns].add_suffix(" до"),
        tuned_df[metric_columns].add_suffix(" после")
    ],
    axis=1
)

for metric in metric_columns:
    comparison_df[f"{metric} изменение"] = (
        comparison_df[f"{metric} после"]
        - comparison_df[f"{metric} до"]
    ).round(4)

display(comparison_df.round(4))


# --- Кодовая ячейка исходного ноутбука: 41 ---
visual_models = {**best_nb_models, **best_models}

classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(11, 8))
fpr_grid = np.linspace(0, 1, 1000)

for model_name, model in visual_models.items():

    model_classes = list(model.classes_)
    probabilities = model.predict_proba(X_test)
    class_indices = [model_classes.index(cls) for cls in classes]
    probabilities = probabilities[:, class_indices]

    tpr_interpolated = []
    for class_idx in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probabilities[:, class_idx])
        tpr_interpolated.append(np.interp(fpr_grid, fpr, tpr))

    mean_tpr = np.mean(tpr_interpolated, axis=0)
    macro_auc = auc(fpr_grid, mean_tpr)
    plt.plot(fpr_grid, mean_tpr, linewidth=2,
             label=f"{model_name} (macro-AUC = {macro_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1,
         label="Случайный классификатор")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Макро-усреднённые ROC-кривые настроенных моделей")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# --- Кодовая ячейка исходного ноутбука: 44 ---
n_models = len(visual_models)
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.ravel()

for ax, (model_name, model) in zip(axes, visual_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)


for ax in axes[n_models:]:
    ax.axis("off")

fig.suptitle("Матрицы ошибок настроенных моделей", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()


# --- Кодовая ячейка исходного ноутбука: 47 ---
np.random.seed(42)
tf.random.set_seed(42)

nn_scaler = StandardScaler()

X_train_nn = nn_scaler.fit_transform(X_train)
X_test_nn = nn_scaler.transform(X_test)

nn_encoder = LabelEncoder()

y_train_nn = nn_encoder.fit_transform(y_train)
y_test_nn = nn_encoder.transform(y_test)

n_features = X_train_nn.shape[1]
n_classes = len(nn_encoder.classes_)


# --- Кодовая ячейка исходного ноутбука: 48 ---
nn_model = Sequential([
    Input(shape=(n_features,)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax")
])

nn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

nn_model.summary()


# --- Кодовая ячейка исходного ноутбука: 50 ---
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

log_root = os.path.join(tempfile.gettempdir(), "siimo_tensorboard_logs")
os.makedirs(log_root, exist_ok=True)

log_dir = os.path.join(
    log_root,
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
os.makedirs(log_dir, exist_ok=True)

tensorboard_cb = TensorBoard(log_dir=log_dir)
print("Папка TensorBoard:", log_dir)


history = nn_model.fit(
    X_train_nn,
    y_train_nn,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, tensorboard_cb],
    verbose=1
)


# --- Кодовая ячейка исходного ноутбука: 52 ---
train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(train_accuracy) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, train_accuracy, label="Обучающая выборка")
axes[0].plot(epochs_range, val_accuracy, label="Валидационная выборка")
axes[0].set_title("Точность нейронной сети")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_range, train_loss, label="Обучающая выборка")
axes[1].plot(epochs_range, val_loss, label="Валидационная выборка")
axes[1].set_title("Функция потерь нейронной сети")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# --- Кодовая ячейка исходного ноутбука: 54 ---
y_pred_proba = nn_model.predict(X_test_nn)

y_pred_nn = np.argmax(y_pred_proba, axis=1)

nn_accuracy = accuracy_score(y_test_nn, y_pred_nn)
nn_precision = precision_score(
    y_test_nn, y_pred_nn, average="weighted"
)
nn_recall = recall_score(
    y_test_nn, y_pred_nn, average="weighted"
)
nn_f1 = f1_score(
    y_test_nn, y_pred_nn, average="weighted"
)

print("Метрики нейронной сети на тестовой выборке:")
print(f"Accuracy:  {nn_accuracy:.4f}")
print(f"Precision: {nn_precision:.4f}")
print(f"Recall:    {nn_recall:.4f}")
print(f"F1-score:  {nn_f1:.4f}")

print("\nОтчёт по классам:")
print(classification_report(
    y_test_nn,
    y_pred_nn,
    target_names=nn_encoder.classes_
))


# --- Кодовая ячейка исходного ноутбука: 57 ---
np.random.seed(42)
tf.random.set_seed(42)


def build_nn_model(hidden_units, activation="relu",
                   learning_rate=1e-3, dropout=0.2):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features,)))

    for i, units in enumerate(hidden_units):
        model.add(Dense(units, activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


experiments = [
    {"name": "Базовая (32-16, relu, lr=1e-3)",
     "hidden": (32, 16), "act": "relu", "lr": 1e-3,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Один слой (64, relu)",
     "hidden": (64,), "act": "relu", "lr": 1e-3,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Три слоя (64-32-16, relu)",
     "hidden": (64, 32, 16), "act": "relu", "lr": 1e-3,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Активация tanh",
     "hidden": (32, 16), "act": "tanh", "lr": 1e-3,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Активация swish",
     "hidden": (32, 16), "act": "swish", "lr": 1e-3,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Learning rate = 1e-2",
     "hidden": (32, 16), "act": "relu", "lr": 1e-2,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Learning rate = 1e-4",
     "hidden": (32, 16), "act": "relu", "lr": 1e-4,
     "batch": 32, "epochs": 50, "dropout": 0.2},

    {"name": "Batch size = 64",
     "hidden": (32, 16), "act": "relu", "lr": 1e-3,
     "batch": 64, "epochs": 50, "dropout": 0.2},

    {"name": "Batch size = 128",
     "hidden": (32, 16), "act": "relu", "lr": 1e-3,
     "batch": 128, "epochs": 50, "dropout": 0.2},

    {"name": "Dropout = 0.3",
     "hidden": (32, 16), "act": "relu", "lr": 1e-3,
     "batch": 32, "epochs": 50, "dropout": 0.3},

    {"name": "Эпох = 20",
     "hidden": (32, 16), "act": "relu", "lr": 1e-3,
     "batch": 32, "epochs": 20, "dropout": 0.2},

    {"name": "Эпох = 100",
     "hidden": (32, 16), "act": "relu", "lr": 1e-3,
     "batch": 32, "epochs": 100, "dropout": 0.2},
]


exp_results = []

for exp in experiments:
    model = build_nn_model(
        hidden_units=exp["hidden"],
        activation=exp["act"],
        learning_rate=exp["lr"],
        dropout=exp["dropout"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=7,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=exp["epochs"],
        batch_size=exp["batch"],
        callbacks=[early_stop],
        verbose=0
    )

    y_proba = model.predict(X_test_nn, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    exp_results.append({
        "Эксперимент": exp["name"],
        "Accuracy": accuracy_score(y_test_nn, y_pred),
        "Precision": precision_score(y_test_nn, y_pred, average="weighted"),
        "Recall": recall_score(y_test_nn, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test_nn, y_pred, average="weighted"),
        "Эпох обучено": len(history.history["loss"]),
        "Val Loss (min)": round(min(history.history["val_loss"]), 4),
    })


df_exp = pd.DataFrame(exp_results)
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    df_exp[col] = df_exp[col].round(4)

print("\nРезультаты экспериментов с гиперпараметрами:")
display(df_exp)


# --- Кодовая ячейка исходного ноутбука: 60 ---
data = [
    ("GaussianNB", .8970, .8997, .8970, .8972, .9899),
    ("MultinomialNB", .8771, .8864, .8771, .8777, .9264),
    ("ComplementNB", .5954, .4758, .5954, .4958, .8761),
    ("BernoulliNB", .7327, .7688, .7327, .6882, .9208),
    ("Decision Tree", .8970, .8971, .8970, .8963, .9693),
    ("LDA", .9033, .9130, .9033, .9046, .9912),
    ("SVM", .9236, .9237, .9236, .9236, .9894),
    ("k-NN", .9158, .9166, .9158, .9160, .9839),
    ("Neural Network", .9240, .9241, .9240, .9240, None),
]

cols = ["Модель", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
df = pd.DataFrame(data, columns=cols).sort_values("F1-score", ascending=False).reset_index(drop=True)

df["ROC-AUC"] = df["ROC-AUC"].apply(lambda v: v if v is not None else "—")
display(df)

fig, ax = plt.subplots(figsize=(14, 7))
x, w = np.arange(len(df)), 0.18
for off, col, c in zip((-2, -1, 0, 1), ("Accuracy", "Precision", "Recall", "F1-score"),
                       ("steelblue", "coral", "seagreen", "goldenrod")):
    ax.bar(x + off * w, df[col], w, label=col, color=c)

roc = df["ROC-AUC"].apply(lambda v: v if v != "—" else 0).astype(float)
ax.bar(x + 2 * w, roc, w, label="ROC-AUC", color="mediumpurple", alpha=0.85)

ax.set(xlabel="Модель", ylabel="Значение метрики", title="Сравнение метрик качества всех моделей",
       ylim=(0.4, 1.05))
ax.set_xticks(x, df["Модель"], rotation=25, ha="right", fontsize=9)
ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)

for i, f1 in enumerate(df["F1-score"]):
    ax.text(i + w, f1 + 0.005, f"{f1:.4f}", ha="center", fontsize=8, color="goldenrod")

plt.tight_layout(); plt.show()
