"""Проверка, что утечка обнаруживается."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Данные
df = pd.read_csv("data/raw/pima-indians-diabetes.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Правильно: scaler обучен на train
scaler_ok = StandardScaler().fit(X_train)
X_train_ok = scaler_ok.transform(X_train)
X_test_ok = scaler_ok.transform(X_test)

# Неправильно: scaler обучен на всех данных (утечка)
scaler_leak = StandardScaler().fit(np.vstack([X_train, X_test]))
X_train_leak = scaler_leak.transform(X_train)
X_test_leak = scaler_leak.transform(X_test)

# Сравните средние train и test
print("OK: mean(train) =", X_train_ok.mean(axis=0).round(3)[:3])
print("OK: mean(test) =", X_test_ok.mean(axis=0).round(3)[:3])
print("LEAK: mean(train) =", X_train_leak.mean(axis=0).round(3)[:3])
print("LEAK: mean(test) =", X_test_leak.mean(axis=0).round(3)[:3])
