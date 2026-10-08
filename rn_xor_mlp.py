import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. DATASET: Tabla de verdad de la puerta XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# 2. CONSTRUCCIÓN Y ENTRENAMIENTO DE LA RED MULTICAPA (MLP)
#    1 capa oculta de 4 neuronas con activación tanh. La capa oculta permite
#    combinar varias rectas y construir una frontera NO lineal, que es lo que
#    el XOR necesita.
red = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh',
                    solver='adam', max_iter=5000, random_state=42)
red.fit(X, y)

# 3. EVALUACIÓN Y PREDICCIONES
predicciones = red.predict(X)

print("--- TABLA DE VERDAD PROBADA POR EL MLP (XOR) ---")
for entrada, objetivo, pred in zip(X, y, predicciones):
    estado = "OK" if objetivo == pred else "ERROR"
    print(f"Entrada: {entrada} -> Esperado: {objetivo} | Predicho: {pred}  [{estado}]")

print(f"\nPrecisión del modelo: {accuracy_score(y, predicciones) * 100}%")
print(f"Iteraciones hasta converger: {red.n_iter_}")

# 4. INSPECCIÓN DE LA ARQUITECTURA APRENDIDA
print("\n--- PARÁMETROS APRENDIDOS POR LA RED ---")
print(f"Pesos Entrada -> Capa Oculta  W(1): {red.coefs_[0].shape}")   # (2, 4)
print(f"Pesos Capa Oculta -> Salida   W(2): {red.coefs_[1].shape}")   # (4, 1)

# A diferencia del perceptrón simple (rn_sencilla_xor.py, que falla al 50%),
# el MLP resuelve el XOR al 100% gracias a su capa oculta no lineal.