import numpy as np
import pandas as pd
import math

def expoy(xi,j):
    y=float((math.pow((xi-meansy[j]),2)/(2*1.0*vary[j])))*(-1)
    h=2*vary[j]
    t=float(y/(1.0*h))
    inter=math.exp(t)
    denom=math.sqrt(2*math.pi)*math.sqrt(vary[j])
    ans=float(inter/(1.0*denom))
    return ans
    
def expon(xi,j):
    y=float((math.pow((xi-meansn[j]),2)/(2*1.0*varn[j])))*(-1)
    h=2*varn[j]
    t=float(y/(1.0*h))
    inter=math.exp(t)
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

acc=0
for i in range(rows):
    py=float(county/(rows*1.0))
    pn=float(countn/(rows*1.0))
    for j in range(cols-1):
        py*=expoy(data[i,j],j)   #Naive bayes probability given yes
        pn*=expon(data[i,j],j)
    if(py>=pn):
        pred='Yes'
        if(data[i,cols-1]==pred):
            acc+=1  #TP
    else:
        pred='No'
        if(data[i,cols-1]==pred):
            acc+=1  #TN

accuracy=float(acc/(rows*1.0))
print "The accuracy is ",100*accuracy


































    
