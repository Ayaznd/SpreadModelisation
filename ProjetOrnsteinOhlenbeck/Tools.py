import numpy as np


def moy(act):
    n = len(act)
    s = 0
    for i in range(n):
        s += act[i]
    return s/n
def std(act):
    n = len(act)
    m = moy(act)
    s = 0
    for i in range(n):
        s += (act[i]-m)**2
    return np.sqrt(s/n)


def cov(act1, act2):
    n, m = len(act1), len(act2)
    m1 ,m2 =moy(act1), moy(act2)
    s=0
    for i in range(n):
        s += act1[i]*act2[i]
    s=s/n-m1*m2
    return s
def corr(act1,act2) :
    std1 , std2 = std(act1) , std(act2)
    coeff=cov(act1,act2)
    coeff=coeff/(std1*std2)
    return coeff

