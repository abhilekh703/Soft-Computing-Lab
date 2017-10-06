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



def buildDataFindAccu():
    accu=[]
    for i in range(30):
        selec=[]
        for j in range(cols-1):
            if(cromo[i,j]==1):
                selec.append(j)
        selec.append(cols-1)
        extractedData=data[:,selec]
        extractedData=np.array(extractedData)
        
        ans=naivebayes(extractedData)
        accu.append(ans)
    accu=np.array(accu)
    #print accu
    return accu


df=pd.read_csv('SPECTF_New.csv')

df=df.sample(frac=1)

data=df.as_matrix()

rows,cols=df.shape
#print rows
#print cols


cromo=[[0]*(cols-1) for i in range(30)]
cromo=np.array(cromo)
for i in range(30):
    for j in range(cols-1):
        f=random.randint(0,1)
        cromo[i,j]=f
"""for i in range(30):
    print cromo[i]"""


accu=buildDataFindAccu()



flag=0
ite=0
while(ite<30):
    flag=0
    print ite," entered"
    for accuCount in range(29):
        if(accu[accuCount]==accu[accuCount+1]):
            flag+=1
    if(flag==29):
        print "Accu for all is same in iteration ",ite
        break          
    
    accuSum=0
    for i in range(30):
        accuSum+=accu[i]
    #print accuSum
    prob=[0.0]*30
    for i in range(30):
        prob[i]=float(accu[i]/(accuSum*1.0))
    #print prob
    s=0
    cummProb=[0.0]*30
    for i in range(30):
        s=s+prob[i]
        cummProb[i]=s
    #print cummProb
    rand=[0.0]*30
    for i in range(30):
        x=random.uniform(0,1)
        rand[i]=x
    #print rand
    chromoSelected = [0.0]*30
    for i in range(30):
        for j in range(30):
            if(cummProb[j]>rand[i]):
                chromoSelected[i]=j
                break
    #print chromoSelected
    countArray=[0]*30
    for i in range(30):
        countArray[chromoSelected[i]]+=1
    #print countArray
    ChromoFive=[-1]*5
    for i in range(5):
        maxi=0
        index=-1
        for j in range(30):
            if(countArray[j]>maxi):
                maxi=countArray[j]
                index=j
        ChromoFive[i]=index
        for k in range(30):
            if(countArray[k]==countArray[index]):
                countArray[k]=-1
    #print ChromoFive
    k=0
    for i in range(30):
        if(k%5==0):
            k=0
        chromoSelected[i]=ChromoFive[k]
        k+=1
    #print chromoSelected
    chromoNew=[[0]*(cols-1) for i in range(30)]
    chromoNew=np.array(chromoNew)
    for i in range(30):
        chromoNew[i]=cromo[chromoSelected[i]]
    cromo=chromoNew
    accu=buildDataFindAccu()
    """print "ChromoNew"
    for i in range(30):
        print cromo[i]"""
  
    crossNo=int(0.25*30)
    #print crossNo
    crossSelected=[-1]*crossNo
    crossSelected=random.sample(range(0,30),crossNo)
    #print crossSelected
    for i in range(crossNo):
        crossPoint=random.randint(1,(cols-1))  
        #print "Cross over point ",crossPoint
        x=cromo[crossSelected[i],0:crossPoint]
        x=np.array(x)
        t=i+1
        if(i==(crossNo-1)):
            t=0
        y=cromo[crossSelected[t],crossPoint::]
        y=np.array(y)
        z=np.concatenate((x,y),axis=0)
        ans=0
        selec=[]
        for j in range(cols-1):
            if(z[j]==1):
                selec.append(j)
        selec.append(cols-1)
        extractedData=data[:,selec]
        extractedData=np.array(extractedData)
        ans=naivebayes(extractedData)
        #print "Previous Accuracy",accu[crossSelected[i]]
        #print "Accuracy after Crossover",ans
        if(ans>accu[crossSelected[i]]):
            cromo[crossSelected[i]]=z
            accu[crossSelected[i]]=ans
        
    print "After Crossover"
    for i in range(30):
        print cromo[i]



    mutateNo=int(0.10*30*(cols-1))
    mutSele=random.sample(range(0,(30*(cols-1))),mutateNo)
    for i in range(mutateNo):
        rn=int(mutSele[i]/(cols-1))
        cn=int(mutSele[i]%(cols-1))
        print "row ",rn," column ",cn
        x=cromo[rn]
        x=np.array(x)
        if(x[cn]==0):
            x[cn]=1
        else:
            x[cn]=0
        ans=0
        selec=[]
        for j in range(cols-1):
            if(x[j]==1):
                selec.append(j)
        selec.append(cols-1)
        extractedData=data[:,selec]
        extractedData=np.array(extractedData)
        ans=naivebayes(extractedData)
        if(ans>accu[rn]):
            accu[cromo[rn]]=ans
            cromo[rn]=x

    print "Chromosomes After Mutation"
    for i in range(30):
        print cromo[i]
    print "accuracy\n",accu
    ite+=1
print "process over"
#print accu
finalMaxaccuIndex=np.argmax(accu)
print "Chromosomes are "

for i in range (0 , 44):
	if(cromo[finalMaxaccuIndex+i]==1):
		print i+1
print cromo[finalMaxaccuIndex,:]
print "accuracy using GA ",accu[finalMaxaccuIndex]


def naiveAccu():
    ans=naivebayes(data)
    return ans


print "Naive bais accuracy"
print naiveAccu()



















