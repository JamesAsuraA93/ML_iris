import numpy as np


def kNN(Xtrain, Ytrain, Xtest, k=1):
    Ytest = []
    for x in Xtest:
        distance = np.sqrt(np.sum((Xtrain - x)**2, axis=1))
        idx = np.argsort(distance)
        (values, counts) = np.unique(Ytrain[idx[:k]], return_counts=True)
        ind = np.argmax(counts)
        Ytest.append(values[ind])
    return Ytest


