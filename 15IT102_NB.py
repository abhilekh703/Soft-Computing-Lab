import numpy as np
import pandas as pd
import math
import random

def naivebayes(extractedData):

    def expoy(xi,j):
        y=float(math.pow((xi-meansy[j]),2))*(-1)
        h=2*vary[j]
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(vary[j])
        ans=float(inter/(1.0*denom))
        return ans
    
    def expon(xi,j):
        y=float(math.pow((xi-meansn[j]),2))*(-1)
        h=2*varn[j]
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(varn[j])
        ans=float(inter/(1.0*denom))
        return ans

    r,c=extractedData.shape
    
    county=0
    countn=0
    for i in range(r):
        if(extractedData[i,c-1]=='Yes'):
            county+=1
        else:
            countn+=1
    meansy=[0.0]*(c-1)
    meansn=[0.0]*(c-1)
    for i in range(c-1):
       for j in range(r):
           if(extractedData[j,c-1]=='Yes'):
               meansy[i]+=data[j,i]
           else:
               meansn[i]+=data[j,i]
    for i in range(c-1):
        meansy[i]=float(meansy[i]/(county*1.0))
        meansn[i]=float(meansn[i]/(countn*1.0))
    vary=[0.0]*(c-1)
    varn=[0.0]*(c-1)
    for i in range(c-1):
        for j in range(r):
            if(extractedData[j,c-1]=='Yes'):
                vary[i]+=math.pow((extractedData[j,i]-meansy[i]),2)
            else:
                varn[i]+=math.pow((extractedData[j,i]-meansn[i]),2)
    for i in range(c-1):
        vary[i]=float(vary[i]/((county-1)*1.0))
        varn[i]=float(varn[i]/((countn-1)*1.0))
    acc=0
    for i in range(r):
        py=float(county/(r*1.0))
        pn=float(countn/(r*1.0))
        for j in range(c-1):
            py*=expoy(extractedData[i,j],j)  
            pn*=expon(extractedData[i,j],j)
        if(py>=pn):
            pred='Yes'
            if(extractedData[i,c-1]==pred):
                acc+=1 
        else:
            pred='No'
            if(extractedData[i,c-1]==pred):
                acc+=1 

    accuracy=float(acc/(r*1.0))
    #print accuracy
    return accuracy



df=pd.read_csv('SPECTF_New.csv')

df=df.sample(frac=1)

data=df.as_matrix()

rows,cols=df.shape
#print rows
#print cols
def naiveAccu():
    ans=naivebayes(data)
    return ans


print "Naive bais accuracy"
print naiveAccu()
