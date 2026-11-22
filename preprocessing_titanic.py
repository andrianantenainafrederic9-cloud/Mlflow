
import pandas as pd

# 1. Charger les donnees
df = pd.read_csv("train.csv")

print("Dimensions initiales :", df.shape)

# 2. Supprimer les colonnes inutiles
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# 3. Remplacer les valeurs manquantes de Age
df["Age"] = df["Age"].fillna(df["Age"].median())

# 4. Remplacer les valeurs manquantes de Embarked
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# 5. Transformer Sex en valeurs numeriques
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# 6. Transformer Embarked en colonnes numeriques
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)

# 7. Verifier les valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# 8. Afficher les donnees preparees
print("\nDonnees preparees :")
print(df.head())

print("\nDimensions finales :", df.shape)

print("\nPretraitement termine !")
