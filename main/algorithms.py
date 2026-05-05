import math
import numpy as np

# KNN
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

# Decision Tree
def gini(groups, classes):
    total_instances = sum([len(group) for group in groups])
    gini_value = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = group.count(class_val) / len(group)
            score += proportion ** 2
        gini_value += (1 - score) * (len(group) / total_instances)
    return gini_value

def split(x, y, feature_index, threshold):
    left_x, left_y, right_x, right_y = [], [], [], []
    for i in range(len(x)):
        if x[i][feature_index] <= threshold:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])
    return left_x, left_y, right_x, right_y

def best_split(X, y):
    classes = list(set(y))
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature_index in range(len(X[0])):
        for i in range(len(X)):
            threshold = X[i][feature_index]
            left_x, left_y, right_x, right_y = split(X, y, feature_index, threshold)
            gini_value = gini([left_y, right_y], classes)
            if gini_value < best_gini:
                best_gini = gini_value
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold

def build_tree(X, y, max_depth=5, depth=0):
    if len(set(y)) == 1:
        return {'leaf': y[0]}

    if depth >= max_depth:
        return {'leaf': max(set(y), key=y.count)}
    
    feature, threshold = best_split(X, y)
    left_x, left_y, right_x, right_y = split(X, y, feature, threshold)
    return {
        'feature_index': feature,
        'threshold': threshold,
        'left': build_tree(left_x, left_y, max_depth, depth + 1),
        'right': build_tree(right_x, right_y, max_depth, depth + 1)
    }

def predict_tree(node, point):
    if 'leaf' in node:
        return node['leaf']
    feature_index = node['feature_index']
    threshold = node['threshold']
    if point[feature_index] <= threshold:
        return predict_tree(node['left'], point)
    else:
        return predict_tree(node['right'], point)
    
def tree_predict(tree, X_test):
    return [predict_tree(tree, point) for point in X_test]

# Logistic Regression
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # np.max для стабильности вычислений
    return exp_z / exp_z.sum()

def predict_proba(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def train_logistic(X, y, learning_rate=0.1, epochs=500):
    n_samples = len(X)
    n_features = len(X[0])
    n_classes = len(set(y))
    
    # начинаем с нулевых весов
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)
    
    X = np.array(X)
    y = np.array(y)
    
    for epoch in range(epochs):
        # 1. предсказываем вероятности для ВСЕХ точек
        probs = np.array([predict_proba(x, W, b) for x in X])
        
        # 2. считаем ошибку — one hot encoding
        # превращаем [0, 1, 2] в [[1,0,0], [0,1,0], [0,0,1]]
        y_onehot = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            y_onehot[i][label] = 1
        
        # 3. считаем градиент (насколько веса неправильные)
        error = probs - y_onehot  

        # 4. обновляем веса — твой код здесь
        W -= learning_rate * np.dot(X.T, error) / n_samples
        b -= learning_rate * error.mean(axis=0)
    
    return W, b

def logistic_predict(X_test, W, b):
    predictions = []
    for x in X_test:
        probs = predict_proba(x, W, b)
        predictions.append(np.argmax(probs))
    return predictions