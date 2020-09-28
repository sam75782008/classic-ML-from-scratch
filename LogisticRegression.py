#ML from scratch
#Logistic Regression

#import package
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 

#logistic regression
class LogisticRegression:
    def __init__(self, lr=0.001, iterations = 1000, fit_intercept=True):
        self.lr = lr
        self.iterations = iterations
        self.fit_intercept = fit_intercept
    
    def add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def fit(self, X, Y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        #initialization beta
        self.beta = np.zeros(X.shape[1]) 
        
        #iteration
        for i in range(self.iterations):
            Z = np.dot(X,self.beta)
            pred = self.sigmoid(Z)
            error = Y-pred
            
            #update beta
            self.beta += self.lr*(np.dot(X.T,error)) #p*1
        return

        
    def predict( self, X ) :

        if self.fit_intercept:
            X = self.add_intercept(X)
            
        return np.round(self.sigmoid(np.dot(X, self.beta)))
    
def main():
    #dataset
    iris = datasets.load_iris()
    X = iris.data[:, :10]
    Y = (iris.target != 0) * 1
       
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                        test_size = 0.2, random_state = 0 )     
    # Model training     
    model = LogisticRegression(lr=0.01, iterations=100, fit_intercept=True) 
    model.fit( X_train, Y_train )
    
    #Model evaluation
    Y_pred = model.predict(X_test)
    print ('Accuracy from scratch:',(Y_pred == Y_test).sum().astype(float) / len(Y_pred))
    
if __name__ == "__main__" :  
    main() 