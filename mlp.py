import _hilos

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

D_DEFAULT = 20
H_DEFAULT = 32
E_DEFAULT = 200
ETA_DEFAULT = 2.0
ALPHA_DEFAULT = 1e-4


def cargar_datos(n, d=D_DEFAULT, test_size=0.2, data_seed=42):
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=max(2, d // 2), n_redundant=d // 4,
        flip_y=0.02, class_sep=1.2, random_state=data_seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=data_seed, stratify=y)
    sc = StandardScaler().fit(X_tr)
    return (
        sc.transform(X_tr).astype(np.float64),
        y_tr.reshape(-1, 1).astype(np.float64),
        sc.transform(X_te).astype(np.float64),
        y_te.reshape(-1, 1).astype(np.float64),
    )


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def entrenar_mlp(seed, X, y, h=H_DEFAULT, E=E_DEFAULT, eta=ETA_DEFAULT, alpha=ALPHA_DEFAULT, B=None):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0.0, np.sqrt(1.0 / d), size=(d, h))
    b1 = np.zeros((1, h))
    W2 = rng.normal(0.0, np.sqrt(1.0 / h), size=(h, 1))
    b2 = np.zeros((1, 1))
    paso = n if not B else B
    for _ in range(E):
        gW1 = np.zeros_like(W1)
        gb1 = np.zeros_like(b1)
        gW2 = np.zeros_like(W2)
        gb2 = np.zeros_like(b2)
        for i in range(0, n, paso):
            Xb = X[i:i + paso]
            yb = y[i:i + paso]
            A1 = _sigmoid(Xb @ W1 + b1)
            A2 = _sigmoid(A1 @ W2 + b2)
            dZ2 = (A2 - yb) * A2 * (1.0 - A2)
            gW2 += A1.T @ dZ2
            gb2 += dZ2.sum(axis=0, keepdims=True)
            dZ1 = (dZ2 @ W2.T) * A1 * (1.0 - A1)
            gW1 += Xb.T @ dZ1
            gb1 += dZ1.sum(axis=0, keepdims=True)
        W1 -= eta * (gW1 / n + alpha * W1)
        b1 -= eta * gb1 / n
        W2 -= eta * (gW2 / n + alpha * W2)
        b2 -= eta * gb2 / n
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "seed": seed}


def accuracy(modelo, X, y):
    A1 = _sigmoid(X @ modelo["W1"] + modelo["b1"])
    y_pred = _sigmoid(A1 @ modelo["W2"] + modelo["b2"])
    return float(np.mean((y_pred >= 0.5) == (y >= 0.5)))


def flops_entrenamiento(n, d=D_DEFAULT, h=H_DEFAULT, E=E_DEFAULT):
    matmul = 2 * n * d * h + 2 * n * h + 2 * n * h + 2 * n * h + 2 * n * d * h
    elementwise = 14 * n * h + 6 * n
    update = 4 * (d * h + h)
    return E * (matmul + elementwise + update)


def modelo_a_vector(modelo):
    return np.concatenate([modelo["W1"].ravel(), modelo["b1"].ravel(), modelo["W2"].ravel(), modelo["b2"].ravel()])


def vector_a_modelo(v, d, h, seed):
    i = 0
    W1 = v[i:i + d * h].reshape(d, h); i += d * h
    b1 = v[i:i + h].reshape(1, h); i += h
    W2 = v[i:i + h].reshape(h, 1); i += h
    b2 = v[i:i + 1].reshape(1, 1)
    return {"W1": W1.copy(), "b1": b1.copy(), "W2": W2.copy(), "b2": b2.copy(), "seed": int(seed)}


def tam_modelo(d=D_DEFAULT, h=H_DEFAULT):
    return d * h + h + h + 1
