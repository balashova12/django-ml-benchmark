import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()

def predict_proba(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def train_logistic(X, y, learning_rate=0.1, epochs=500):
    n_samples = len(X)
    n_features = len(X[0])
    n_classes = len(set(y))
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)
    X = np.array(X)
    y = np.array(y)
    for epoch in range(epochs):
        probs = np.array([predict_proba(x, W, b) for x in X])
        y_onehot = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            y_onehot[i][label] = 1
        error = probs - y_onehot
        W -= learning_rate * np.dot(X.T, error) / n_samples
        b -= learning_rate * error.mean(axis=0)
    return W, b

def logistic_predict(X_test, W, b):
    predictions = []
    for x in X_test:
        probs = predict_proba(x, W, b)
        predictions.append(np.argmax(probs))
    return predictions