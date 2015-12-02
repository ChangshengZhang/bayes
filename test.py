#!/usr/bin/python
# File Name: test.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com
# Created Time: Thu Nov 26 01:22:40 2015

#########################################################################
import numpy as np
import scipy.stats as stats
import scipy.special as spec
from rpy2.robjects.packages import GeneralizedHyperbolic


def clac(c,d,h):
    mode = (h-1+np.sqrt((h-1)**2+1.0*c/d))/c
    x = np.linspace(1,2*max(mode,20),200)
    result = []
    for item in x:
        f = (1.0*c/d)**(1.0*h/2)*item**(h-1)*np.exp(-0.5*(c*item+1.0*d/item))
        g = stats.gamma.pdf(item,mode+1)*2*spec.kv(h,np.sqrt(c*d))
        print f*1.0/g
        result.append(f*1.0/g)
    return max(result)

if __name__ == "__main__":

    c = np.linspace(1,1000,2000)
    d = np.linspace(1,1000,2000)
    h = np.linspace(1,1000,2000)
    
    max_num = 0
    for cc in c:
        for dd in d:
            for hh in h:
                temp = clac(cc,dd,hh)
                if temp > max_num:
                    max_num = temp
            print "hh"

    print max_num
