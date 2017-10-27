import numpy as np
import pandas as pd
import math
import random

df=pd.read_csv('SPECTF_New.csv')
#df=df.sample(frac=1)
data=df.as_matrix()
rows,cols=df.shape
classlabel=data[0::,cols-1]
#print classlabel
data=data[0::,0:cols-1]
#print data
print '\nnumber of rows ',rows,'number of cols ',cols, '\n'

k=2
c1=[]
c2=[]
selected=random.sample(range(0,rows),k)
print 'first selected',selected[0]
print 'second selected',selected[1]

for i in range(cols-1):
    c1.append(data[selected[0],i]*1.0)

for i in range(cols-1):
    c2.append(data[selected[1],i]*1.0)

c1=np.array(c1)
c2=np.array(c2)
print 'c1 is ',c1
print 'c2 is ',c2

clus1=[]
clus2=[]
flag=0
ite=0
while(ite<50):
    clus1=[]
    clus2=[]
    for i in range(rows):
        d1=0
        d2=0
        for j in range(cols-1):
            d1+=math.pow(data[i,j]-c1[j],2)
            d2+=math.pow(data[i,j]-c2[j],2)
        d1=float(math.sqrt(d1))
        d2=float(math.sqrt(d2))
        #print 'd1 ',d1,' d2 ',d2
        if(d1<=d2):
            clus1.append(i)
        else:
            clus2.append(i)
    
    print 'clus1 array is ',clus1
    print 'clus2 array is ',clus2
    clus1=np.array(clus1)
    clus2=np.array(clus2)
    #print len(c1)
    #print len(c2)
    
    ans1=[]
    ans2=[]
    
    for i in range(cols-1):
        s1=0
        for j in range(len(clus1)):
            s1+=data[clus1[j],i]
        s1=float(s1/len(clus1))
        ans1.append(s1)
        s2=0
        for j in range(len(clus2)):
            s2+=data[clus2[j],i]
        s2=float(s2/len(clus2))
        ans2.append(s2)
    
    ans1=np.array(ans1)
    ans2=np.array(ans2)
    
    print 'c1 now ',ans1,' c1 previous ',c1
    print 'c2 now ',ans2,' c2 previous ',c2

    print '\ncluster 1 size is ',len(clus1),' cluster 2 size is ',len(clus2),'\n'
    cou1=0
    cou2=0
    k=0
    for i in range(cols-1):
        if(c1[i]==ans1[i]):
            cou1+=1
        else:
            c1[i]=ans1[i]
            
    for i in range(cols-1):
        if(c2[i]==ans2[i]):
            cou2+=1
        else:
            c2[i]=ans2[i]
            
    if(cou1==(cols-1) and cou2==(cols-1)):
        break
    cou1=0
    cou2=0
    ite+=1

print len(clus1)
print len(clus2)
yclu1=0
nclu1=0
clusp1=''
clusp2=''
acc=0
for i in range(len(clus1)):
    #print classlabel[clus1[i]]
    if(classlabel[clus1[i]]=='Yes'):
        yclu1+=1
    else:
        nclu1+=1
print 'yes in cluster1 ',yclu1,' no in cluster1 ',nclu1
if(yclu1>=nclu1):
    print "cluster1 have a cluster of yes"
    clusp1='Yes'
else:
    print "cluster1 have a cluster of no"
    clusp1='No'

for i in range(len(clus1)):
    if(classlabel[clus1[i]]==clusp1):
        acc+=1

yclu2=0
nclu2=0
for i in range(len(clus2)):
    #print classlabel[clus1[i]]
    if(classlabel[clus2[i]]=='Yes'):
        yclu2+=1
    else:
        nclu2+=1
print 'yes in cluster2 ',yclu2,' no in cluster2 ',nclu2
if(yclu2>=nclu2):
    print "cluster2 have a cluster of yes"
    clusp2='Yes'
else:
    print "cluster2 have a cluster of no"
    clusp2='No'

for i in range(len(clus2)):
    if(classlabel[clus2[i]]==clusp2):
        acc+=1

print "accuracy of Kmeans",float(acc/(rows*1.0))*100
