import math

def distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def get_neighbors(X_train, y_train, new_point, k):
    neighbors = []
    for i in range(len(X_train)):
        dist = distance(X_train[i], new_point)
        neighbors.append((dist, y_train[i]))
    neighbors.sort()
    return neighbors[:k]

def predict_one(neighbors):
    classes = {}
    for dist, k in neighbors:
        if k in classes:
            classes[k] += 1
        else:
            classes[k] = 1
    return max(classes, key=classes.get)

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for new_point in X_test:
        neighbors = get_neighbors(X_train, y_train, new_point, k)
        predicted_class = predict_one(neighbors)
        predictions.append(predicted_class)
    return predictions