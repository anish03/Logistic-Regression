import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
plt.style.use('fivethirtyeight')

datafile = 'datafile.txt'
data = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)

X = np.transpose(np.array(data[:-1]))
Y = np.transpose(np.array(data[-1:]))

pos = np.array([X[i] for i in xrange(X.shape[0]) if Y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if Y[i] == 0])

def plot_data():
    #plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='k',cmap=plt.cm.Paired)
    plt.plot(pos[:,0],pos[:,1],'wo',label='Admitted')
    plt.plot(neg[:,0],neg[:,1],'bo',label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.grid(True)

y_arr = data[-1:]
y_arr = y_arr.flatten()
#print y_arr.flatten()
#print y_arr

#step value for creating meshgrid
h = 0.5

logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X,y_arr)

x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z = logreg.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,8))
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)


plt.xticks(())
plt.yticks(())

plot_data()
plt.show()
