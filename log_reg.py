import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy import optimize
plt.style.use('dark_background')
import pandas as pd


datafile = 'ex2data1.txt'

data = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)

X = np.transpose(np.array(data[:-1]))
y = np.transpose(np.array(data[-1:]))

#size of data
m = y.size
X = np.insert(X,0,1,axis=1)

#cases where the student gets admitted
pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i]==1])

#cases where the student does not get admitted
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i]==0])

def plot_data():
    plt.figure(figsize=(10,8))
    plt.plot(pos[:,1],pos[:,2],'ro',label='Admitted')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend()
    plt.grid(True)
    #plt.show()

plot_data()

#logistic regression
def hypothesis(theta,X):
    return expit(np.dot(X,theta))

#calculate the logistic regression cost function
def computeCost(theta,X,y,_lambda=0.):
    term1 = np.dot(-np.array(y).T,np.log(hypothesis(theta,X)))
    term2 = np.dot((1-np.array(y)).T,np.log(1-hypothesis(theta,X)))
    reg_term = (_lambda/2) * np.sum(np.dot(theta[1:].T,theta[1:]))
    return float((1./m) * (np.sum(term1 - term2 + reg_term)))

initial_theta = np.zeros((X.shape[1],1))

computeCost(initial_theta,X,y)

#
def optimizeTheta(theta,X,y,_lambda=0.):
    result = optimize.fmin(computeCost,x0=theta,args=(X,y,_lambda),maxiter=400,full_output=True)
    return result[0],result[1]


theta, minCost = optimizeTheta(initial_theta,X,y)

def predict(theta,X,threshold=0.5):
    p = expit(np.dot(X,theta.T)) >= threshold
    return p.astype('int')

p = predict(theta,X)
print 'Training accuracy is',(100*sum(p == y.ravel())/p.size)

#Plotting the decision boundary
boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])

#slope of the decision line
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
plt.plot(boundary_xs,boundary_ys,'m-',label='Decision Boundary')
plt.plot(40,85,'wo')
plt.show()


