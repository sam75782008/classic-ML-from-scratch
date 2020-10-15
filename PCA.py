# ML from scratch
#Principal Component Analysis, PCA

#packages
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class PCA():
    def __init__(self,k):
        self.component = k
        
    
    def covariance(self,X,Y=None):
        n = X.shape[0]
        X = X - np.mean(X,axis=0)
        Y = X if not Y else Y - np.mean(Y,axis=0)
        return (1/n)*np.matmul(X.T,Y)
    
    def fit(self,X):
        
        #covariance matrix
        cov = self.covariance(X)
        
        #eigen decomponsition
        eigenvalue, eigenvector = np.linalg.eig(cov)
        
        #sort and select the eigen vector
        idx = np.argsort(eigenvalue[::-1])
        eigenvector = eigenvector[:,idx]
        eigenvector = eigenvector[:,:self.component]
        
        return np.matmul(X,eigenvector)

def main():
    #dataset
    digits = load_digits()
    X = digits.data
    print("Input data size:",X.shape)

    # PCA reduce dimension
    pca = PCA(k=2)
    projected_X = pca.fit(X)
    print("After dimension reduction:",projected_X.shape)
    #prediction

    #plot
    plt.figure(figsize=(8,6))
    plt.scatter(projected_X[:, 0], projected_X[:, 1],
                c=digits.target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();

    
if __name__ == "__main__" :  
    main() 