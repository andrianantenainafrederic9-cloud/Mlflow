
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

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

mlflow.set_experiment("Titanic_Prediction")


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

df = pd.read_csv("train.csv")

print("Dimensions du dataset :", df.shape)


# ==========================================
# 3. SELECTION DES VARIABLES
# ==========================================

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
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==========================================
# 5. CREATION DU MODELE
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])


# ==========================================
# 6. SEPARATION TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Donnees entrainement :", len(X_train))
print("Donnees test :", len(X_test))


# ==========================================
# 7. ENTRAINEMENT ET SUIVI MLFLOW
# ==========================================

with mlflow.start_run(run_name="RandomForest_Titanic"):

    # PARAMETRES
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    # ENTRAINEMENT
    model.fit(X_train, y_train)

    # PREDICTIONS
    y_pred = model.predict(X_test)

    # METRIQUES
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ENREGISTREMENT DES METRIQUES
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # RAPPORT DE CLASSIFICATION
    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    mlflow.log_artifact("classification_report.txt")

    # ENREGISTREMENT DU MODELE
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )

    # AFFICHAGE DES RESULTATS
    print("\n========== RESULTATS ==========")

    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1-score :", f1)

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    print("\nModele enregistre avec succes dans MLflow !")

