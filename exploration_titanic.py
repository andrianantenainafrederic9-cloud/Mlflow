
import pandas as pd

# Charger le dataset
df = pd.read_csv("train.csv")

# Afficher les 5 premières lignes
print("\n--- PREMIERES LIGNES ---")
print(df.head())

# Dimensions du dataset
print("\n--- DIMENSIONS ---")
print(df.shape)

# Informations générales
print("\n--- INFORMATIONS ---")
df.info()

# Valeurs manquantes
print("\n--- VALEURS MANQUANTES ---")
print(df.isnull().sum())

# Statistiques descriptives
print("\n--- STATISTIQUES ---")
print(df.describe())

# Distribution de la variable cible
print("\n--- SURVIVED ---")
print(df["Survived"].value_counts())

print("\nExploration terminee !")
