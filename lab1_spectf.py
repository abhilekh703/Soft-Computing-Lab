import pandas as pd
import numpy as np

doc = pd.read_csv("rand.csv")
doc = doc.sample(frac=1)

alp = 0.1
y = []

X = doc.as_matrix()
X = X[:, :44]
n = 45
X = np.insert(X, 44, 1, axis=1)
np.random.shuffle(X)
# To make the y class
for item in doc['Class']:
    if item == 0:
        y.append(0)
    else:
        y.append(1)

weights = [float(1/n)]*45
weights = np.array(weights)

def get_kfold(k):
    ip_test = []
    ip_train = []
    op_train = []
    op_test = []
    for i in range(0, len(X)):
        if i % 10 == k:
            ip_test.append(X[i])
            op_test.append(y[i])
        else:
            ip_train.append(X[i])
            op_train.append(y[i])
    return np.array(ip_train), np.array(ip_test), np.array(op_train), np.array(op_test)


def predict(ip_train, i):
    ip_train = np.array(ip_train)
    if np.sum(np.multiply(ip_train[i], weights)) > 0:
        return 1
    else:
        return 0
overall_accuracy = 0
for i in range(10):
    total_sum = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    num_positive = 0
    num_negative = 0
    ip_train, ip_test, op_train, op_test = get_kfold(i)
    for _ in range(500):
        for j in range(90):
            y_pred = predict(ip_train, j)
            weights = [x + alp*ip_train[j][ind]*(op_train[j]-y_pred) for ind, x in enumerate(weights)]
    total_right = 0
    for t in range(10):
        z = predict(ip_test, t)
        if z == op_test[t]:
            total_right += 1
        if z == 0 and op_test[t] == 0:
            true_negative += 1
        if z == 0 and op_test[t] == 1:
            false_positive += 1
        if z == 1 and op_test[t] == 1:
            true_positive += 1
        if z == 1 and op_test[t] == 0:
            false_negative += 1
    accuracy = total_right / 10.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print("Fold " + str(i) + " produced an accuracy of " + str(accuracy) + " Precision: " + str(precision) + " Recall: " + str(recall))
    accuracy = total_right/10.0
    total_sum += accuracy