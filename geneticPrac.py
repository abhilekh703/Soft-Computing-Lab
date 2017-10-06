import numpy as np
import pandas as pd
import math
import random

def expoy(xi,j):
    y=float((math.pow((xi-meansy[j]),2)/(2*1.0*vary[j])))*(-1)
    # h=2*vary[j]
    # t=float(y/(1.0*h))
    inter=math.exp(y)
    denom=math.sqrt(2*math.pi)*math.sqrt(vary[j])
    ans=float(inter/(1.0*denom))
    return ans
    
def expon(xi,j):
    y=float((math.pow((xi-meansn[j]),2)/(2*1.0*varn[j])))*(-1)
    # h=2*varn[j]
    # t=float(y/(1.0*h))
    inter=math.exp(y)
    denom=math.sqrt(2*math.pi)*math.sqrt(varn[j])
    ans=float(inter/(1.0*denom))
    return ans

#Read fron file
df=pd.read_csv('SPECTF_New.csv')
#Shuffle the dataset
df=df.sample(frac=1)
#Convert the dataset to a matrix
data=df.as_matrix()
#Return dimensions of a matrix
rows,cols=df.shape
#print rows
#print cols
chro=np.random.randint(2, size=(30, 44))
#print(np.matrix(chro))
county=0  #Number of yes in dataset
countn=0  #Number of no in dataset
#print data[:,cols-1]
for i in range(rows):
    if(data[i,cols-1]=='Yes'):
        county+=1
    else:
        countn+=1
# print 'county ',county,' countn ',countn
meansy=[0.0]*(cols-1)  #Find mean for each column given class label as yes
meansn=[0.0]*(cols-1)  
for i in range(cols-1):
    for j in range(rows):
        if(data[j,cols-1]=='Yes'):
            meansy[i]+=data[j,i]
        else:
            meansn[i]+=data[j,i]
for i in range(cols-1):
    meansy[i]=float(meansy[i]/(county*1.0))
    meansn[i]=float(meansn[i]/(countn*1.0))
vary=[0.0]*(cols-1) #find variance for each column given yes
varn=[0.0]*(cols-1)
for i in range(cols-1):
    for j in range(rows):
        if(data[j,cols-1]=='Yes'):
            vary[i]+=math.pow((data[j,i]-meansy[i]),2)
        else:
            varn[i]+=math.pow((data[j,i]-meansn[i]),2)
for i in range(cols-1):
    vary[i]=float(vary[i]/((county-1)*1.0))
    varn[i]=float(varn[i]/((countn-1)*1.0))

for l in range(40):
	acc=np.zeros(30)
	for k in range(30):
		for i in range(rows):
		    py=float(county/(rows*1.0))
		    pn=float(countn/(rows*1.0))
		    for j in range(cols-1):
		    	if(chro[k,j]==0):
		    		py=py*1.0
		    		pn=pn*1.0
		    	else:
			        py*=expoy(data[i,j],j)   #Naive bayes probability given yes
			        pn*=expon(data[i,j],j)
		    if(py>=pn):
		        pred='Yes'
		        if(data[i,cols-1]==pred):
		            acc[k]+=1  #TP
		    else:
		        pred='No'
		        if(data[i,cols-1]==pred):
		            acc[k]+=1  #TN
		print 'Old Accuracy',float(acc[k]/(rows*1.0))
	tot_fit=0
	for i in range(30):
		tot_fit=tot_fit+acc[i]

	pro=np.zeros(30)
	cum=np.zeros(30)
	for i in range(30):
		pro[i]=acc[i]/tot_fit
		if(i==0):
			cum[i]=pro[i]
		else:
			cum[i]=cum[i-1]+pro[i]
	# for i in range(20):
	# 	print 'Accuracy  ',acc[i],'Probability  ',pro[i],'Cumulative  ',cum[i]
	ran=[0.0]*30
	sele=np.zeros(30)
	for i in range(30):
		ran[i]=random.random()
	for i in range(30):
		for j in range(30):
			if(ran[i]<cum[j]):
				sele[i]=j
				break
		# print sele[i]

	chronew=np.random.randint(2, size=(30, 44))
	for i in range(30):
		for j in range(44):
			t=int(sele[i])
			chronew[i,j]=chro[t,j]
	# print sele[0]
	# print(np.matrix(chro))
	# print '\n\n\nHello\n\n\n'
	# print(np.matrix(chronew))
	# for i in range(30):
	# for j in range(44):
	# 	print chronew[0,j]
	crossover_pt=30;
	crossarr=random.sample(range(0,30), 8)      #Unique random numbers
	# for i in range(8):
	# 	print crossarr[i]

	chronew2=np.random.randint(2, size=(30, 44))
	for i in range(30):
		for j in range(44):
			chronew2[i,j]=chronew[i,j]
	# for j in range(44):
	# 	print chronew2[0,j], chronew[0,j]
	for i in range(8):
		for j in range(44):
			if(j>=crossover_pt):
				if(i==7):
					chronew2[crossarr[i],j]=chronew[crossarr[0],j]
				else:
					chronew2[crossarr[i],j]=chronew[crossarr[i+1],j]
	#nutationrate = 10%  0.1*1320=132

	mutx=[0.0]*132
	muty=[0.0]*132
	for i in range(132):
		mutx[i]=random.randint(0,29)
		muty[i]=random.randint(0,43)
	for i in range(132):
		if(chronew2[mutx[i],muty[i]]==0):
			chronew2[mutx[i],muty[i]]=1;
		else:
			chronew2[mutx[i],muty[i]]=0;
	for i in range(30):
		for j in range(44):
			chronew[i,j]=chronew2[i,j]
	acc=np.zeros(30)
	for k in range(30):
		for i in range(rows):
		    py=float(county/(rows*1.0))
		    pn=float(countn/(rows*1.0))
		    for j in range(cols-1):
		    	if(chronew[k,j]==0):
		    		py=py*1.0
		    		pn=pn*1.0
		    	else:
			        py*=expoy(data[i,j],j)   #Naive bayes probability given yes
			        pn*=expon(data[i,j],j)
		    if(py>=pn):
		        pred='Yes'
		        if(data[i,cols-1]==pred):
		            acc[k]+=1  #TP
		    else:
		        pred='No'
		        if(data[i,cols-1]==pred):
		            acc[k]+=1  #TN
		print 'New Accuracy',float(acc[k]/(rows*1.0))  
	chro=chronew
	# print(np.matrix(chronew))
	# print(np.matrix(chronew2))

