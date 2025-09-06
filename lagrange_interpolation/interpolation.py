import numpy as np
import math

feats = [(-1, 0), (0, 0), (1, 1)]
feats.sort(key = lambda tup: tup[0])
x, y = ([feat[0] for feat in feats], [feat[1] for feat in feats])


#int_pol = np.polynomial.Polynomial([0])
#for k in range(len(feats)):
#    p = np.polynomial.Polynomial([1])
#    for j in range(len(feats)):
#        if k != j:
#            p *= np.polynomial.Polynomial([-x[j], 1]) / (x[k] - x[j])
#    p *= y[k]
#    int_pol += p
#
#print(int_pol)

int_pol = sum([p for p in 
               [math.prod([np.polynomial.Polynomial([-x[j], 1]) / (x[k] - x[j])
                if j != k else 1 for j in range(len(feats))]) * y[k]
                for k in range(len(feats))]
                ])
print(int_pol)
