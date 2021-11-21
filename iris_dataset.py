# import dataset iris

import pandas as pd
import numpy as np
import os

def load(path='./iris.csv',split_train_test=None):
    if os.path.isfile(path):
        iris = pd.read_csv(path)
    else:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = pd.read_csv(url,header=None)
        iris.to_csv(path,index=False)
    X = iris.iloc[:,:4].values
    y = iris.iloc[:,-1].values
    if split_train_test is not None:
        classes = np.unique(y)
        itrain = np.empty((0,),dtype=np.int)
        itest = np.empty((0,),dtype=np.int)
        for i in classes:
            idx = np.where(y==i)[0]
            split = int(len(idx)*split_train_test)
            itrain = np.concatenate((itrain,idx[:split]))
            itest = np.concatenate((itest,idx[split:]))
        return X[itrain],y[itrain],X[itest],y[itest]
    return X,y

if __name__ == '__main__':
    irisInputs, irisTargets = load()
    print(irisInputs)
    print(irisTargets)