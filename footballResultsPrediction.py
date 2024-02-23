import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("results.csv")
print(df.head())
le = LabelEncoder()

#Variable dependiente
df["ganador"] = df["home_score"] > df["away_score"]

#Variables independientes
X = df[["home_team", "away_team"]]
y = df["ganador"]


#Cofidicacion
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

#Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train,y_train)

prediciones = model.predict(X_test)
precision = accuracy_score(y_test,prediciones)
matriz_confusion = confusion_matrix(y_test, prediciones)

#Comprobar precision del modelo
print("Precision del modelo: ", precision)
print("Matriz de confusion: \n", matriz_confusion)

paises = df["country"].unique()
print("Paises disponibles:")
for i in range(0, len(paises), 4):
    for j in range(i, min(i + 4, len(paises))):
      print(f"{paises[j]:<15}             ", end="")
    print()

paisLocal = input("Introduce el pais local: ")
paisVisitante = input("Introduce el pais visitante: ")

predicion_df = pd.DataFrame({"home_team": [paisLocal], "away_team": [paisVisitante]})
prediccion_encode = encoder.transform(predicion_df)

probabilidades = model.predict_proba(prediccion_encode)
porcentaje = probabilidades[0][1]

print(f"Probabilidad de victoria del {paisLocal}: {porcentaje:.2%}")
print(f"Probabilidad de victoria del {paisVisitante}: {(1-porcentaje)*100:.2f}% ")