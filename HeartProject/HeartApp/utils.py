
from math import *
#i=0
def hist(sx):
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1        #(age,target):count   ex: (52,0):15 times repeated
    #global i
    #if i==0:
    #    print(d)
    return map(lambda z: float(z)/len(sx), d.values())    #[15/1025]=

def elog(x):
    #print(x,end='\t')
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)   #0.014634146341463415*log(0.014634146341463415)

def cmidd(x, y, z): #mutual information estimator
    return entropyd(list(zip(y, z)))+entropyd(list(zip(x, z)))-entropyd(list(zip(x, y, z)))-entropyd(z)

def entropyd(sx, base=2):
    #global i
    #if i==0:
    #    print(len(sx))  len(sx)=1025
    #    print(sx)         (attibute,target) i.e ex: (52,0),.....

    return entropyfromprobs(hist(sx), base=base)

def entropyfromprobs(probs, base=2):
    #global i
    #if i==0:
    #    print("hello")
    #    print(list(probs))
    return -sum(map(elog, probs))/log(base)

def midd(x, y):   #mutual information estimator
    #global i
    #i=0
    t= -entropyd(list(zip(x, y)))+entropyd(x)+entropyd(y)
    #t1= -entropyd(list(zip(x, y)))
    #print(t1)
    #t2=entropyd(x)
    #print(t2)
    #t3=entropyd(y)
    #print(t3)
    #i+=1
    return t
