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

def split(X, y, feature_index, threshold):
    left_x, left_y, right_x, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_index] <= threshold:
            left_x.append(X[i])
            left_y.append(y[i])
        else:
            right_x.append(X[i])
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
    if point[node['feature_index']] <= node['threshold']:
        return predict_tree(node['left'], point)
    else:
        return predict_tree(node['right'], point)

def tree_predict(tree, X_test):
    return [predict_tree(tree, point) for point in X_test]