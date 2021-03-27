import numpy as np
from sys import stdin
from sklearn.neural_network import MLPClassifier
import pickle


mlp = pickle.load(open('mlp2.sav', 'rb'))

# classify terminal input
for line in stdin:
    if line == '': 
        break
    d=np.array([float(x) for x in line.split(',')])
    
    #preprocessing
    d = d[-201:]
    d[1:] -= d[:-1]
    d = np.delete(d, [0])
    d[d < 0] = 0
    d[d > 0] = 1

    print(int(mlp.predict([d])))
