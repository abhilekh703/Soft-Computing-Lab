import numpy as np
import pandas as pd

doc = pd.read_csv("IRIS.csv")
doc = doc.sample(frac=1)

alp = 0.1
y = []
n = 5

#print(doc)
X = doc.as_matrix()
#print(X)
X = X[:, :4]
print(X.shape)
X = np.insert(X, 4, 1, axis=1)
print(X.shape)
# To make the y class
for item in doc['class']:
    if item == "Iris-setosa":
        y.append(0)
    else:
        y.append(1)

wts = [float(1/n)]*5
print(wts)
wts = np.array(wts)
print(wts)

def get_kfold(k):
    ip_test = []
    ip_train = []
    op_train = []
    op_test = []
    for i in range(0, 100):
        if i % 10 == k:
            ip_test.append(X[i])
            op_test.append(y[i])
        else:
            ip_train.append(X[i])
            op_train.append(y[i])
    return np.array(ip_train), np.array(ip_test), np.array(op_train), np.array(op_test)

def predict(ip_train, i):
    ip_train = np.array(ip_train)
    #print(ip_train[i], wts)
    if np.sum(np.multiply(ip_train[i], wts)) > 0:
        return 1
    else:
        return 0

total_sum = 0

for i in range(10):
    ip_train, ip_test, op_train, op_test = get_kfold(i)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negatve = 0
    for j in range(90):
        y_pred = predict(ip_train, j)
        wts = [x + alp*ip_train[j][ind]*(op_train[j]-y_pred) for ind, x in enumerate(wts)]
    total_right = 0
    for xx in range(10):
        z = predict(ip_test, xx)
        #print(op_test)
        if z == op_test[xx]:
            total_right += 1
        if z==0 and op_test[xx]==0:
            true_negative+=1
        if z==0 and op_test[xx]==1:
            false_positive+=1
        if z==1 and op_test[xx]==1:
            true_positive+=1
        if z==1 and op_test[xx]==0:
            false_negatve+=1
    accuracy = total_right/10.0
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negatve)
    print("Fold " + str(i) + " produced an accuracy of "+str(accuracy)+" Precision: "+str(precision)+" Recall: "+str(recall))
    total_sum += accuracy

print("Average accuracy is: "+str(total_sum/10.0))