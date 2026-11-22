
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ==================================================
# 1. CONFIGURATION MLFLOW
# ==================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Titanic_Comparaison_Modeles")


# ==================================================
# 2. CHARGEMENT DU DATASET
# ==================================================

df = pd.read_csv("train.csv")

print("Dimensions du dataset :", df.shape)


# ==================================================
# 3. SELECTION DES VARIABLES
# ==================================================

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


# ==================================================
# 4. PREPARATION DES DONNEES
# ==================================================

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
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==================================================
# 5. SEPARATION TRAIN / TEST
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Donnees entrainement :", len(X_train))
print("Donnees test :", len(X_test))


# ==================================================
# 6. DEFINITION DES MODELES
# ==================================================

models = {

    "Random_Forest": {
        "classifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        "parameters": {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
    },

    "Logistic_Regression": {
        "classifier": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "parameters": {
            "max_iter": 1000,
            "random_state": 42
        }
    },

    "Decision_Tree": {
        "classifier": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "parameters": {
            "max_depth": 5,
            "random_state": 42
        }
    },

    "XGBoost": {
        "classifier": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ),
        "parameters": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
}


# ==================================================
# 7. TYPES AUTORISES POUR LA SAUVEGARDE
# ==================================================

trusted_types = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier"
]


# ==================================================
# 8. ENTRAINEMENT ET COMPARAISON
# ==================================================

results = []

for model_name, model_config in models.items():

    print("\n===================================")
    print("Entrainement :", model_name)
    print("===================================")

    classifier = model_config["classifier"]

    with mlflow.start_run(run_name=model_name):

        # Creation du pipeline
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        # Entrainement
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Calcul des metriques
        accuracy = accuracy_score(y_test, y_pred)

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

        # Enregistrement du modele et des parametres
        mlflow.log_param("model", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_params(model_config["parameters"])

        # Enregistrement des metriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Rapport de classification
        report = classification_report(
            y_test,
            y_pred
        )

        report_filename = f"classification_report_{model_name}.txt"

        with open(report_filename, "w", encoding="utf-8") as file:
            file.write(report)

        mlflow.log_artifact(report_filename)

        # Sauvegarde du modele dans MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=trusted_types
        )

        # Conservation des resultats
        results.append({
            "Modele": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

        # Affichage
        print("Accuracy :", accuracy)
        print("Precision :", precision)
        print("Recall :", recall)
        print("F1-score :", f1)


# ==================================================
# 9. CLASSEMENT DES MODELES
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print("\n========== CLASSEMENT DES MODELES ==========")

print(results_df.to_string(index=False))


# ==================================================
# 10. IDENTIFICATION DU MEILLEUR MODELE
# ==================================================

best_model = results_df.iloc[0]

print("\n========== MEILLEUR MODELE ==========")

print("Modele :", best_model["Modele"])
print("Accuracy :", best_model["Accuracy"])
print("Precision :", best_model["Precision"])
print("Recall :", best_model["Recall"])
print("F1-score :", best_model["F1-score"])


# ==================================================
# 11. EXPORT DES RESULTATS
# ==================================================

results_df.to_csv(
    "comparaison_resultats.csv",
    index=False
)

print("\nResultats enregistres dans comparaison_resultats.csv")

print("\nComparaison terminee avec succes !")
