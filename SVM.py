# ML from scratch
# Support vector machine

#import package
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 

#SVM
class SVM:
    def __init__(self, lamb = 0.01, lr=0.001, iterations = 1000, fit_intercept=True):
        self.lr = lr
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.lamb = lamb
    
    def add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    
    def fit(self, X, Y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        #initialization beta
        self.beta = np.zeros((X.shape[1],1)) 
        Y = Y.reshape(Y.shape[0],1)
        
        #iteration
        for i in range(self.iterations):
            pred = np.dot(X,self.beta) #n*1
            err = np.multiply(Y,pred)<1 #n*1
            indiactor = np.multiply(err,Y) #n*1
            repeat_matrix = np.repeat(indiactor,X.shape[1],axis=1) #n*p
            dbeta = np.dot(np.ones((1,Y.shape[0])), np.multiply(repeat_matrix,X))/Y.shape[0] #1*p
            
            #update beta
            self.beta = self.beta + self.lr*dbeta.T #[p,1]
            self.beta[1:X.shape[1]] = self.beta[1:X.shape[1]] - self.lamb * self.beta[1:X.shape[1]] #regularization
            
        return

        
    def predict( self, X ) :

        if self.fit_intercept:
            X = self.add_intercept(X)
            
        y_pred = np.dot(X,self.beta)
        pred = []
        for val in y_pred:
            if(val > 1):
                pred.append(1)
            else:
                pred.append(-1)        
            
        return pred
    
def main():
    #dataset
    iris = datasets.load_iris()
    X = iris.data[:, :10]
    Y = (iris.target != 0) * 1
    Y = 2*Y-1 #{-1, 1}
       
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                        test_size = 0.2, random_state = 0 )     
    # Model training     
    model = SVM(lamb = 0.01, lr=0.1, iterations=200, fit_intercept=True) 
    model.fit( X_train, Y_train )
    
    #Model evaluation
    Y_pred = model.predict(X_test)
    print ('Accuracy from scratch:',(Y_pred == Y_test).sum().astype(float) / len(Y_pred))
    
if __name__ == "__main__" :  
    main() 