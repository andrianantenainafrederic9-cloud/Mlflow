
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================
# 1. CONFIGURATION MLFLOW
# ==========================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Titanic_Hyperparameter_Tuning")


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

df = pd.read_csv("train.csv")

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]

print("Dimensions du dataset :", df.shape)


# ==========================================
# 3. SEPARATION TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Donnees entrainement :", X_train.shape[0])
print("Donnees test :", X_test.shape[0])


# ==========================================
# 4. PRETRAITEMENT DES DONNEES
# ==========================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


numeric_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer(transformers=[
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])


# ==========================================
# 5. PIPELINE DU MODELE
# ==========================================

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(max_iter=2000)
    )
])


# ==========================================
# 6. HYPERPARAMETRES A TESTER
# ==========================================

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__solver": [
        "lbfgs",
        "liblinear"
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


# ==========================================
# 7. VALIDATION CROISEE
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# ==========================================
# 8. ENTRAINEMENT ET TRACKING MLFLOW
# ==========================================

with mlflow.start_run(
    run_name="Logistic_Regression_Tuning"
):

    print("\nDebut de l'optimisation...\n")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_

    best_cv_score = grid_search.best_score_


    # Affichage des meilleurs parametres
    print("\n===== MEILLEURS PARAMETRES =====")

    print(best_params)

    print("Meilleur F1 CV :", best_cv_score)


    # Enregistrement des parametres
    mlflow.log_params(best_params)

    mlflow.log_metric(
        "best_cv_f1",
        best_cv_score
    )


    # ==========================================
    # 9. EVALUATION SUR LES DONNEES DE TEST
    # ==========================================

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print("\n===== RESULTATS FINAUX =====")

    print("Accuracy :", accuracy)

    print("Precision :", precision)

    print("Recall :", recall)

    print("F1-score :", f1)


    # Enregistrement des metriques
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    })


    # ==========================================
    # 10. RAPPORT DE CLASSIFICATION
    # ==========================================

    report = classification_report(
        y_test,
        y_pred
    )

    print("\n===== CLASSIFICATION REPORT =====")

    print(report)


    with open(
        "tuning_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


    mlflow.log_artifact(
        "tuning_classification_report.txt"
    )


    # ==========================================
    # 11. SAUVEGARDE DU MEILLEUR MODELE
    # ==========================================

    trusted_types = [
        "numpy.dtype"
    ]

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_logistic_model",
        skops_trusted_types=trusted_types
    )


    print("\nMeilleur modele enregistre dans MLflow.")


# ==========================================
# 12. FIN
# ==========================================

print("\nOptimisation terminee avec succes !")
