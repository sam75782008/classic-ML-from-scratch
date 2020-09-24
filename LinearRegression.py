# Code from scratch: Linear Regression
# This script is a practice to develop
# Linear Regression from scratch
# Date: 09/23/2020

#import packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

#read data
data = pd.read_excel('data/energy_data.xlsx')
print(data.info())

#separte data
x = pd.concat([data['AT'],data['V'],data['AP'],data['RH']],axis=1).values
y = data['PE'].values

#standardize data
sc = StandardScaler()
x = sc.fit_transform(x)

#define MSE
def cost_function(X, Y, beta):
    m = len(Y)
    J = np.sum((X.dot(beta)-Y)**2)/(2*m)
    
    return J

#gradient descent
def batch_gradient_descent(X, Y, beta, learning_rate, iterations):
    cost_history = [0]*iterations
    m = len(Y)
    
    for iteration in range(iterations):
        h = X.dot(beta) #nx1
        loss = h - Y #nx1
        gradient = X.T.dot(loss) / m #px1
        beta = beta - learning_rate * gradient
        cost = cost_function(X, Y, beta)
        cost_history[iteration] = cost
        
    return beta, cost_history

#experiment
m = 7000
X_train = x[:m]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
Y_train = y[:m]

X_test = x[m:]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
Y_test = y[m:]

#initialize beta
beta = np.zeros(X_train.shape[1])

#train
learning_rate = 0.01
iterations = 2000
new_beta, cost_history = batch_gradient_descent(X_train, Y_train, beta, 
                                                learning_rate, iterations)


#plot
plt.figure(figsize=(12,9))
plt.plot(cost_history)
plt.ylabel('Loss', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#r2
def r2(y_pred, y_true):
    sst = np.sum((y_true-y_true.mean())**2)
    ssr = np.sum((y_pred-y_true)**2)
    
    r2 = 1-(ssr/sst)
    
    return(r2)
    
#evaluation
print('R2 score:',r2(X_test.dot(new_beta),Y_test))
