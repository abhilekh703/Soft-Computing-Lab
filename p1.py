"""import csv
with open('IRIS.csv','rb') as f:
	reader=csv.reader(f)
	for row in reader:
		print row"""
import pandas as pd
import numpy as np
df=pd.read_csv('IRIS.csv')
"""print df.tail()
for row in df:
	print row"""
y = []

for i in df['class']:
    if i == "Iris-setosa":
        y.append(0)
    else:
        y.append(1)
# Now we have the y array
x = df.as_matrix()
print x
print 'numpy array starts'
x = np.array(x)
print x
x = np.delete(x, -1, axis=1)
x = np.insert(x, 4, 1, axis=1)

# Now we have the feature matrix
print(x)
y = np.array(y)
print y
print 'y begins'
y = y.transpose()
print y
weights = [float(1.0/(x.shape[1]+1))]*5
weights = np.array(weights)
print(weights)
print x.size
print x.shape
print x.shape[1]
def predict(x_val, y_val):
    prod = np.multiply(x_val, weights)
    if prod > 0:
    	y1.append(1)
    else:
    	y1.append(0)
    y1=np.array(y1)
    print (y1-y_val)
    #print(np.sum(prod)-y_val)

for feature, ans in zip(x, y):
    predict(feature, ans)
