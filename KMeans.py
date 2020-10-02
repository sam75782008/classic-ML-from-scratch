#ML from scratch

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class KMeans():
    def __init__(self, numb_cluster, max_iterations, tolerance):
        self.K = numb_cluster
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def euclidean(self, X1, X2):
        diff = X1 - X2
        return np.sqrt(np.dot(diff,diff.T))
    
    def fit(self, X):
        
        #initialization
        centroid = X[:self.K,:]
        pred = np.zeros((len(X),1))
        
        #iteration
        for iteration in range(self.max_iterations):
            #prediction
            for i in range(len(X)):
                distances = [self.euclidean(X[i,:],centro) for centro in centroid]
                cluster =  distances.index(min(distances))
                pred[i] = cluster
            
            #collect classes
            classes = {}
            for i in range(self.K):
                classes[i] = []
            for i in range(len(X)):
                classes[int(pred[i])].append(X[i,:])
            
            #calculate new centroid
            temp = []
            for i in range(self.K):
                temp.append(np.average(classes[i],axis=0))
            
            #check tolerance
            find = True
            if np.sum((np.array(temp) - np.array(centroid))/np.array(centroid) * 100.0) > self.tolerance:
                find = False
            centroid = temp
            
            if find:
                break
            
        
        return pred, centroid

def main():
    #dataset
    df = pd.read_csv('data\\cluster_validation_data.txt', sep=",", header=None)
    df.head()

    # normalize data
    X = df.values
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    # Model   
    model = KMeans(numb_cluster=3, max_iterations=500, tolerance =1e-5) 
    Y_pred, centroid = model.fit(X)
    #prediction
    plt.figure(figsize=(15,10))
    plt.scatter(X[:,0],X[:,1],c=Y_pred)
    for i in range(len(centroid)):
    	plt.scatter(centroid[i][0], centroid[i][1], s = 500, marker = "x")
    plt.show()

    
if __name__ == "__main__" :  
    main() 
            
            
            
            
            