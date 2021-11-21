from kNN import kNN
import iris_dataset
import numpy as np
import matplotlib.pyplot as plt

def plotdata(Xtrain, ytrain, Xtest=[], ytest=[],Ztest=[]):
    color = {
        'Iris-setosa': 'b',
        'Iris-versicolor': 'g',
        'Iris-virginica': 'r'
    }
    for i in range(len(Xtrain)):
        plt.plot(Xtrain[i][0], Xtrain[i][1],'x' ,c=color[ytrain[i]],mfc='none')
    for i in range(len(Xtest)):
        plt.plot(Xtest[i][0], Xtest[i][1],'.' ,c='none',mfc=color[ytest[i]])
    for i in range(len(Ztest)):
        plt.plot(Xtest[i][0], Xtest[i][1],'o' ,c=color[Ztest[i]],mfc='none')



if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = iris_dataset.load(split_train_test=0.5)

    plt.figure(1)
    rate = []
    K = range(1,len(Xtrain)+1)
    for k in K:
        Ztest = kNN(Xtrain, ytrain, Xtest, k)
        plotdata(Xtrain, ytrain, Xtest, ytest,Ztest)
        plt.title('kNN: k=%d' % k)
        plt.draw()
        plt.pause(0.001)
        plt.cla()
        rate.append(np.sum(Ztest == ytest)/len(ytest) * 100)
    plt.figure(2)
    plt.plot(K, rate)
    plt.xlabel('k')
    plt.axis([0,80,30,100])
    plt.ylabel('accuracy')
    plt.title('kNN: accuracy')
    plt.show()
    print('accuracy:',rate)

    plt.figure(3)
    k = rate.index(max(rate)) + 1
    Ztest = kNN(Xtrain, ytrain, Xtest, k)
    plotdata(Xtrain, ytrain, Xtest, ytest,Ztest)
    plt.title('kNN: k=%d' % k)
    plt.show()
