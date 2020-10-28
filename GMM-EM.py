#ML from scratch
# Gaussian Mixture Model-Expectation Maxization

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

#GMM
class GMM():
    def __init__(self,k,num_iter):
        self.k = k
        self.iter = num_iter
    
    def initialize(self,X):
        self.shape = X.shape
        self.n, self.m = self.shape
        
        #phi (prob of category i)
        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        
        #wij (prob of xi from catgory j)
        self.weights = np.full(shape=(self.n, self.k), fill_value=1/self.k)
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        #mean of each class
        self.mu = [X[row_index,:] for row_index in random_row]
        #sigma
        self.sigma = [np.cov(X.T) for _ in range(self.k)]
    
    def e_step(self,X):
        # E-Step: update weights and phi holding mu and sigma constant
        self.weights = self.predict_prob(X)
        self.phi = self.weights.sum(axis=0)
    
    def m_step(self,X):
        # M-Step: update mu and sigma holding phi and weights constant
        for i in range(self.k):
            weight = self.weights[:,[i]]
            total_weight = weight.sum()
            self.mu[i] = (X*weight).sum(axis=0)/total_weight
            self.sigma[i] = np.cov(X.T, aweights=(weight/total_weight).flatten())
    
    def predict_prob(self,X):
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            distribution = multivariate_normal(
                    mean = self.mu[i],
                    cov = self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
            
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:,np.newaxis]
        weights = numerator / denominator
        return weights
    
    def fit(self,X):
        self.initialize(X)
        
        for iteration in range(self.iter):
            self.e_step(X)
            self.m_step(X)
    
    def predict(self,X):
        weights = self.predict_prob(X)
        return np. argmax(weights, axis=1)

def main():
    iris = load_iris()
    X = iris.data
    np.random.seed(42)
    gmm = GMM(k=3, num_iter=50)
    gmm.fit(X)
    

if __name__=='__main__':
    main()