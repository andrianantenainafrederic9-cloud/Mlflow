
import mlflow
from mlflow import MlflowClient

# ==========================================
# 1. CONFIGURATION MLFLOW
# ==========================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient(
    tracking_uri="http://127.0.0.1:5000"
)

# ==========================================
# 2. IDENTIFIANT DU RUN
# ==========================================

run_id = "b0cfe5b72d514bfba94d8c7f6194f363"

model_uri = f"runs:/{run_id}/best_logistic_model"

# ==========================================
# 3. NOM DU MODELE
# ==========================================

model_name = "Titanic_Logistic_Regression"

# ==========================================
# 4. ENREGISTREMENT DANS LE REGISTRY
# ==========================================

print("Enregistrement du modele dans le Registry...")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("\nModele enregistre avec succes !")

print("Nom :", model_version.name)
print("Version :", model_version.version)
print("Statut :", model_version.status)

# ==========================================
# 5. DESCRIPTION DU MODELE
# ==========================================

client.update_model_version(
    name=model_name,
    version=model_version.version,
    description=(
        "Modele de prediction de survie Titanic. "
        "Logistic Regression optimisee avec GridSearchCV "
        "et validation croisee en 5 plis."
    )
)

# ==========================================
# 6. AJOUT DES TAGS
# ==========================================

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="algorithm",
    value="LogisticRegression"
)

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="dataset",
    value="Titanic"
)

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="f1_score",
    value="0.7260"
)

# ==========================================
# 7. AFFICHAGE DES INFORMATIONS
# ==========================================

print("\n===== INFORMATIONS DU REGISTRY =====")

model_details = client.get_model_version(
    name=model_name,
    version=model_version.version
)

print("Nom du modele :", model_details.name)
print("Version :", model_details.version)
print("Description :", model_details.description)
print("Run ID :", model_details.run_id)

print("\nTags :")

for key, value in model_details.tags.items():
    print(key, ":", value)

print("\nModel Registry termine avec succes !")
