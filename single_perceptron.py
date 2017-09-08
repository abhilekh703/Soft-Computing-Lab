import pandas as pd
import numpy as np

df=pd.read_csv('IRIS.csv')
df=df.sample(frac=1)
#print df.head
X=df.as_matrix()
y=X[:,4]
X=X[:,0:4]
X=np.insert(X, 4, 1, axis=1)
y = [0 if x == 'Iris-setosa'  else 1 for x in y]
n=X.shape[1]
#print n
weights = [float(1.0/n)]*5
#print weights
#alpha=0.1
alpha=input("Enter learning rate: ")
threshold=0

def kfold(k):
	x_test=[]
	x_train=[]
	y_test=[]
	y_train=[]
	for i in range(0,len(X)):
		if i%10==k:
			x_test.append(X[i])
			y_test.append(y[i])
		else:
			x_train.append(X[i])
			y_train.append(y[i])
	return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def predict_output(dataset,row):
	out=np.multiply(dataset[row],weights)
	out=np.sum(out)
	if out>threshold:
		return 1
	else:
		return 0	

sum=0
for k in range(0,10):
	X_train, X_test, y_train, y_test=kfold(k)
	weights=[(float)(1.0/n)]*5
	for i in range(0,90):
		out=predict_output(X_train,i)
		weights=[x+alpha*(y_train[i]-out)*X_train[i][ind] for ind,x in enumerate(weights)]
	true=0
	for i in range(0,10):
		z=predict_output(X_test,i)
		yi=y_test[i]
		if z==yi:
			true=true+1
	accuracy=(float)(true)/10.0;
	print "Fold "+str(k)+" Accuracy: "+str(accuracy)
	sum+=accuracy		
		
print "Mean accuracy is "+str((float)(sum)/10.0)
		
		
		
		
		
