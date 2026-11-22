
import mlflow
import mlflow.sklearn
import pandas as pd

# 1. Connexion au serveur MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Identifiant du Run déjà exécuté
run_id = "abb02f091e95420784f2e3d04fc07647"

# 3. Adresse du modèle enregistré
model_uri = f"runs:/{run_id}/random_forest_model"

# 4. Charger le modèle sans le réentraîner
model = mlflow.sklearn.load_model(model_uri)

print("Modele charge avec succes !")

# 5. Créer les informations d'un nouveau passager
nouveau_passager = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}])

# 6. Effectuer la prédiction
prediction = model.predict(nouveau_passager)

# 7. Afficher le résultat
if prediction[0] == 1:
    print("Le passager pourrait survivre.")
else:
    print("Le passager pourrait ne pas survivre.")

# 8. Afficher les probabilités
probabilites = model.predict_proba(nouveau_passager)

print("\nProbabilites :")
print("Non-survie :", probabilites[0][0])
print("Survie :", probabilites[0][1])
