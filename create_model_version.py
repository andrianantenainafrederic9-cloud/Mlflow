
import mlflow
from mlflow import MlflowClient

# Connexion au serveur MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Nom du modele deja enregistre
model_name = "Titanic_Logistic_Regression"

# Identifiant du run contenant le modele
run_id = "b0cfe5b72d514bfba94d8c7f6194f363"

# Chemin du modele
model_uri = f"runs:/{run_id}/best_logistic_model"

# Creer une nouvelle version
version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("Nouvelle version creee !")
print("Nom :", model_name)
print("Version :", version.version)
print("Statut :", version.status)

# Ajouter une description
client.update_model_version(
    name=model_name,
    version=version.version,
    description="Version de demonstration pour apprendre la gestion des versions MLflow."
)

# Ajouter des tags
client.set_model_version_tag(
    name=model_name,
    version=version.version,
    key="type",
    value="demonstration"
)

client.set_model_version_tag(
    name=model_name,
    version=version.version,
    key="dataset",
    value="Titanic"
)

print("Description et tags ajoutes !")
