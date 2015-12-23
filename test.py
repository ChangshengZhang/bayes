

import numpy as np
matrix = np.array([[1,0],[0,1]])
flag =1

try:
    test = np.linalg.cholesky(matrix)
    flag =0
except:
    print "not positive define."

print test
print "flag",flag


