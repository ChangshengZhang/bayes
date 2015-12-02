#!/usr/bin/python
# File Name: rgig.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com
# Created Time: Thu Nov 26 03:19:29 2015

#########################################################################

from rpy2.robjects.packages import importr 
import rpy2.robjects as robjects

def GigSample(b,c,d):
    rgig = importr("GeneralizedHyperbolic")

    robjects.r('''
        f<- function(b,c,d){
            return(rgig(1,b,c,d))
        }
        ''')

    return robjects.r['f'](b,c,d)

if __name__ == "__main__":
    a = GigSample(2,2,2)
    print a[0]
