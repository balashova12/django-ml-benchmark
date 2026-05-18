from django.shortcuts import render
from django.http import JsonResponse
import time
import tracemalloc
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from .algorithms.knn import knn_predict
from .algorithms.decision_tree import build_tree, tree_predict
from .algorithms.logistic_regression import train_logistic, logistic_predict
import numpy as np

iris = datasets.load_iris()
X = iris.data.tolist()
y = iris.target.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def index(request):
    return render(request, 'index.html')

def run_algorithm(request, algorithm):
    tracemalloc.start()
    start = time.perf_counter()

    if algorithm == 'knn':
        predictions = knn_predict(X_train, y_train, X_test)

    elif algorithm == 'knn_lib':
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).tolist()

    elif algorithm == 'tree':
        tree = build_tree(X_train, y_train)
        predictions = tree_predict(tree, X_test)

    elif algorithm == 'tree_lib':
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).tolist()

    elif algorithm == 'logistic':
        W, b = train_logistic(X_train, y_train)
        predictions = logistic_predict(X_test, W, b)

    elif algorithm == 'logistic_lib':
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).tolist()

    else:
        return JsonResponse({'error': 'Алгоритм не найден'}, status=404)

    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    accuracy = accuracy_score(y_test, predictions)

    return JsonResponse({
        'algorithm': algorithm,
        'time': round(elapsed, 6),
        'memory_kb': round(memory / 1024, 2),
        'accuracy': round(float(accuracy), 4),
    })