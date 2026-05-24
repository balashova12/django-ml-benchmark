# django-ml-benchmark

A benchmarking tool that compares manually implemented machine learning algorithms against scikit-learn, with an interactive Django web interface.

## Overview

This project investigates performance differences between ML algorithms written from scratch and their scikit-learn equivalents — measuring execution time, memory usage, and prediction accuracy. Understanding these differences helps developers make informed decisions and gain deeper insight into how these algorithms work under the hood.

## Algorithms Implemented

| Algorithm | Custom Implementation | scikit-learn Equivalent |
|---|---|---|
| K-Nearest Neighbors | ✅ | `KNeighborsClassifier` |
| Decision Tree | ✅ | `DecisionTreeClassifier` |
| Logistic Regression | ✅ | `LogisticRegression` |

## Features

- Custom implementations of three classic ML algorithms built without ML libraries
- Side-by-side performance comparison against scikit-learn
- Django REST endpoints returning JSON results — no page reload required
- Interactive charts (Chart.js) comparing execution time per algorithm
- Results table showing execution time (s), memory usage (KB), and accuracy
- Responsive layout that adapts to different screen sizes

## Tech Stack

- **Backend:** Python, Django (MVT pattern)
- **ML:** NumPy, scikit-learn (for comparison only)
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Dataset:** Iris dataset (150 samples, 70/30 train-test split)

## How It Works

1. User clicks a button to run a custom or scikit-learn algorithm
2. Browser sends a GET request to an endpoint like `/run/knn/`
3. Django view runs the algorithm, measures time and memory, returns JSON
4. JavaScript updates the results table and bar charts without reloading the page

## Project Structure

```
practice/
├── manage.py                  — entry point
├── myproject/
│   ├── settings.py            — project configuration
│   └── urls.py                — main router
└── main/
    ├── algorithms/
    │   ├── knn.py
    │   ├── decision_tree.py
    │   └── logistic_regression.py
    ├── static/css/
    │   └── style.css
    ├── templates/
    │   └── index.html
    ├── views.py               — request handling & benchmarking logic
    └── urls.py                — app routes
```

## Getting Started

### Prerequisites

```
Python 3.10+
```

### Installation

```bash
git clone https://github.com/balashova12/django-ml-benchmark.git
cd django-ml-benchmark
pip install -r requirements.txt
python manage.py runserver
```

Then open `http://127.0.0.1:8000` in your browser.

## Results

All three algorithms achieve 95–100% accuracy on the Iris dataset, confirming correctness of the custom implementations. Performance differences between manual and library versions are visualized in the bar charts.
