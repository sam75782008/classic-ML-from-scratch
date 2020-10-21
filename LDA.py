# ML from scratch
#Linear Discriminant Analysis

#packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#Fisher
class Fisher_LDA():
    def __init__(self,X,Y,K):
        self.X = X
        self.Y = Y
        self.K = K
        self.groupbyClass()
        self.calculate_means()
        self.calculate_Sb_Sw()
    
    def groupbyClass(self):
        self.group_data = {}
        for i in range(len(self.Y)):
            if self.Y[i] not in self.group_data:
                self.group_data[self.Y[i]] = [self.X[i,:]]
            else:
                self.group_data[self.Y[i]].append(self.X[i,:])
        
        self.num_class = len(self.group_data)
    
    def calculate_means(self):
        self.mean_per_class = {}
        self.overall_mean = np.mean(self.X, axis = 0)
        for i in range(self.num_class):
            if i not in self.mean_per_class:
                self.mean_per_class[i] = np.mean(self.group_data[i],axis=0)
    
    def calculate_Sb_Sw(self):
        #Sb
        self.Sb = np.zeros((self.X.shape[1],self.X.shape[1]))
        for i in range(self.num_class):
            mk_minus_m = self.mean_per_class[i]-self.overall_mean
            self.Sb += np.dot((mk_minus_m.T*self.num_class),mk_minus_m) #pxp
        
        #Sw
        self.Sw = np.zeros((self.X.shape[1],self.X.shape[1]))
        for i in range(self.num_class):
            mk = self.mean_per_class[i]
            for j in self.group_data[i]:
                xnk_minus_mk = j-mk
                self.Sw += np.dot(xnk_minus_mk.T,xnk_minus_mk)
    
    def calculate_eigen(self):
        mat = np.dot(np.linalg.pinv(self.Sw),self.Sb)
        
        #eigen decomponsition
        eigenvalue, eigenvector = np.linalg.eig(mat)
        
        #sort and select the eigen vector
        idx = np.argsort(eigenvalue[::-1])
        eigenvector = eigenvector[:,idx]
        self.W = eigenvector[:,:self.K]
        return self.W

def main():
    iris = datasets.load_iris()
    X = iris.data[:, :10]
    Y = iris.target
    print("Input data size:",X.shape)
    print("Input data label:",Y.shape)

    # PCA reduce dimension
    W = Fisher_LDA(X,Y,K=2).calculate_eigen()
    projected_X =np.dot(X,W)
    print("After dimension reduction:",projected_X.shape)
    #prediction
    
    
    #plot
    plt.figure(figsize=(8,6))
    plt.scatter(projected_X[:, 0], projected_X[:, 1],
                c=iris.target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();

    
if __name__ == "__main__" :  
    main() 