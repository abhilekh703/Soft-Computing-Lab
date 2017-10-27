import numpy as np
import pandas as pd
import math
import random

def finddist(x,y):
    s=0
    for i in range(cols-1):
        s+=math.pow(x[i]-y[i],2)
    s=float(math.sqrt(s))
    return s
    
df=pd.read_csv('SPECTF_New.csv')
#df=df.sample(frac=1)
data=df.as_matrix()
rows,cols=df.shape
classlabel=data[0::,cols-1]
print classlabel
data=data[0::,0:cols-1]
print data
print '\nrows ',rows,' cols ',cols, '\n'

c=2
c1=[]
c2=[]
m=raw_input("Enter fuzziness parameter\n")
#print m
m=float(m)

selected=random.sample(range(0,rows),c)
print selected[0]
print selected[1]

for i in range(cols-1):
    c1.append(data[selected[0],i])

for i in range(cols-1):
    c2.append(data[selected[1],i])

c1=np.array(c1)
c2=np.array(c2)
#print 'c1 is ',c1
#print 'c2 is ',c2
#weights1=np.empty(shape=[0,5])
#u=np.empty(shape=[rows,2])
flag=0
ite=0
u=[]
while(ite<50):
    u=[]
    for i in range(rows):
        t=[-1.0]*2
        u.append(t)
    u=np.array(u)
    #print u
    for i in range(rows):
        ans=-1
        xi=data[i,0::]
        cj=c1
        dis_xi_cj=finddist(xi,cj)
        #print dis_xi_cj
        if(dis_xi_cj==0):
            u[i,0]=1
        deno=0
        for it in range(2):
            if(it==0):
                cj=c1
            else:
                cj=c2
            dt=finddist(xi,cj)
            #print dt
            if(dt==0):
                u[i,0]=0
            if(dt!=0):
                par=float(dis_xi_cj/(dt*1.0))
                po=float(2/((m-1)*1.0))
                deno+=float(math.pow(par,po))
        if(deno!=0):
            ans=float(1/(deno*1.0))
            print ans
            u[i,0]=ans
        u[i,1]=1-u[i,0]
    
    print u

    clu1=[]
    clu2=[]
    for i in range(cols-1):
        num1=0
        den1=0
        num2=0
        den2=0
        for j in range(rows):
            num1+=math.pow(u[j,0],m)*data[j,i]
            den1+=math.pow(u[j,0],m)
            num2+=math.pow(u[j,1],m)*data[j,i]
            den2+=math.pow(u[j,1],m)
        ans1=float(num1/(den1*1.0))
        ans2=float(num2/(den2*1.0))
        clu1.append(ans1)
        clu2.append(ans2)

    clu1=np.array(clu1)
    clu2=np.array(clu2)

    cou1=0
    cou2=0

    for i in range(cols-1):
        if(c1[i]==clu1[i]):
            cou1+=1
        else:
            c1[i]=clu1[i]
        if(c2[i]==clu2[i]):
            cou2+=1
        else:
            c2[i]=clu2[i]

    if(cou1==(cols-1) and cou2==(cols-1)):
        break
            
    #print clu1
    #print clu2
    ite+=1

print 'over'
print u

cluster1=[]
cluster2=[]

for i in range(rows):
    if(u[i,0]>=u[i,1]):
        cluster1.append(i)
    else:
        cluster2.append(i)

print len(cluster1)
print len(cluster2)

acc=0
clusp1=''
clusp2=''
yclu1=0
nclu1=0

for i in range(len(cluster1)):
    if(classlabel[cluster1[i]]=='Yes'):
        yclu1+=1
    else:
        nclu1+=1
print 'yclu1 ',yclu1,' nclu1 ',nclu1
if(yclu1>=nclu1):
    print "clus1 have a cluster of yes"
    clusp1='Yes'
else:
    print "clus1 have a cluster of no"
    clusp1='No'
for i in range(len(cluster1)):
    if(classlabel[cluster1[i]]==clusp1):
        acc+=1

yclu2=0
nclu2=0
for i in range(len(cluster2)):
    #print classlabel[clus1[i]]
    if(classlabel[cluster2[i]]=='Yes'):
        yclu2+=1
    else:
        nclu2+=1
print 'yclu2 ',yclu2,' nclu2 ',nclu2
if(yclu2>=nclu2):
    print "clus2 have a cluster of yes"
    clusp2='Yes'
else:
    print "clus2 have a cluster of no"
    clusp2='No'

for i in range(len(cluster2)):
    if(classlabel[cluster2[i]]==clusp2):
        acc+=1

print "accuracy ",float(acc/(rows*1.0))*100



        




