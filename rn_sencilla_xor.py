import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. DATASET: Tabla de verdad de la puerta XOR (OR exclusivo)
#    Devuelve 1 cuando las entradas son DISTINTAS.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# 2. CONSTRUCCIÓN Y ENTRENAMIENTO DE LA NEURONA
neurona = Perceptron(max_iter=100, random_state=42)
neurona.fit(X, y)

# 3. EVALUACIÓN Y PREDICCIONES
predicciones = neurona.predict(X)

print("--- TABLA DE VERDAD PROBADA POR LA NEURONA (XOR) ---")
for entrada, objetivo, pred in zip(X, y, predicciones):
    estado = "OK" if objetivo == pred else "ERROR"
    print(f"Entrada: {entrada} -> Esperado: {objetivo} | Predicho: {pred}  [{estado}]")

print(f"\nPrecisión del modelo: {accuracy_score(y, predicciones) * 100}%")

# 4. INSPECCIÓN DE PESOS APRENDIDOS
pesos = neurona.coef_[0]
sesgo = neurona.intercept_[0]

print("\n--- PARÁMETROS APRENDIDOS POR LA NEURONA ---")
print(f"Peso W1 (para Entrada 1): {pesos[0]}")
print(f"Peso W2 (para Entrada 2): {pesos[1]}")
print(f"Sesgo (Bias b): {sesgo}")

# IMPORTANTE: el XOR NO es linealmente separable. Las clases quedan en
# esquinas opuestas y ninguna recta puede separarlas, por eso el perceptrón
# se estanca en 50% con W=[0,0], b=0. No es un error de código: es el límite
# teórico del perceptrón simple. La solución es un Perceptrón Multicapa (MLP),
# ver rn_xor_mlp.py