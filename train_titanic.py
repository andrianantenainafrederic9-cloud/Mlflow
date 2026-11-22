
import mlflow

# Nom de l'experience
mlflow.set_experiment("Titanic_Prediction")

# Demarrer une execution
with mlflow.start_run(run_name="Premier_Test"):

    # Parametres
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # Metriques de test
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("precision", 0.82)
    mlflow.log_metric("recall", 0.80)

    print("Experience enregistree dans MLflow !")
