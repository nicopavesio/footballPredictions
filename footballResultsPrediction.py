import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
<<<<<<< HEAD
from sklearn.preprocessing import OneHotEncoder
=======
>>>>>>> 8c706e0e8a06781389a9efd7cb60ab696cd3330a


df = pd.read_csv("results.csv")
print(df.head())
le = LabelEncoder()

#Variable dependiente
df["ganador"] = df["home_score"] > df["away_score"]

#Variables independientes
<<<<<<< HEAD
X = df[["home_team", "away_team"]]
y = df["ganador"]


#Cofidicacion
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

#Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

=======
df["equipo_visitante"] = df["away_team"]
df["equipo_local"] = df["home_team"]
df["equipo_local_encoded"] = le.fit_transform(df["equipo_local"])
df["equipo_visitante_encoded"] = le.fit_transform(df["equipo_visitante"])

X_train, X_test, y_train, y_test = train_test_split(df[["equipo_local_encoded", "equipo_visitante_encoded"]], df["ganador"], test_size=0.1)
>>>>>>> 8c706e0e8a06781389a9efd7cb60ab696cd3330a

model = LogisticRegression()
model.fit(X_train,y_train)

prediciones = model.predict(X_test)
precision = accuracy_score(y_test,prediciones)
matriz_confusion = confusion_matrix(y_test, prediciones)

#Comprobar precision del modelo
print("Precision del modelo: ", precision)
print("Matriz de confusion: \n", matriz_confusion)
<<<<<<< HEAD
=======
print("Número de filas para entrenar:", len(X_train))
>>>>>>> 8c706e0e8a06781389a9efd7cb60ab696cd3330a

paises = df["country"].unique()
print("Paises disponibles:")
for i in range(0, len(paises), 4):
    for j in range(i, min(i + 4, len(paises))):
      print(f"{paises[j]:<15}             ", end="")
    print()

paisLocal = input("Introduce el pais local: ")
paisVisitante = input("Introduce el pais visitante: ")
<<<<<<< HEAD

predicion_df = pd.DataFrame({"home_team": [paisLocal], "away_team": [paisVisitante]})
prediccion_encode = encoder.transform(predicion_df)

probabilidades = model.predict_proba(prediccion_encode)
porcentaje = probabilidades[0][1]

print(f"Probabilidad de victoria del {paisLocal}: {porcentaje:.2%}")
print(f"Probabilidad de victoria del {paisVisitante}: {(1-porcentaje)*100:.2f}% ")
=======
local = True
paisLocal_encoded = le.transform([paisLocal]).reshape(-1, 1)
paisVisitante_encoded = le.transform([paisVisitante]).reshape(-1, 1)

porcentaje = model.predict(pd.DataFrame([[paisLocal_encoded[0][0], paisVisitante_encoded[0][0]]], columns=["equipo_local_encoded", "equipo_visitante_encoded"]))

print(f"Probabilidad de victoria del {paisLocal}: {porcentaje[0]:.2%}")
print(f"Probabilidad de victoria del {paisVisitante}: {1-porcentaje[0]:.2%}")
>>>>>>> 8c706e0e8a06781389a9efd7cb60ab696cd3330a
