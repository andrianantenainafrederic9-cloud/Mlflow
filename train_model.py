
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les donnees
df = pd.read_csv("train.csv")

# 2. Selectionner les variables
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

# 3. Transformer les variables manquantes et textuelles
X = X.copy()

X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

X = pd.get_dummies(X, columns=["Sex", "Embarked"], dtype=int)

# 4. Diviser les donnees
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Creer le modele
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# 6. Entrainer le modele
model.fit(X_train, y_train)

# 7. Faire des predictions
y_pred = model.predict(X_test)

# 8. Evaluer le modele
accuracy = accuracy_score(y_test, y_pred)

print("Nombre de donnees d'entrainement :", len(X_train))
print("Nombre de donnees de test :", len(X_test))

print("\nAccuracy :", accuracy)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

print("\nEntrainement termine !")
