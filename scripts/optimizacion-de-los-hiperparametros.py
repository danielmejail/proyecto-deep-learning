from sklearn.model_selection import train_test_split

# Reservamos, del conjunto de train, un subconjunto de validación
# para realizar la optimización de los hiperparámetros del modelo
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.2,
    random_state=42
)
