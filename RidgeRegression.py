#ML from scratch
#Ridge Regression

#import package
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

class Ridge_Regression():
    
    def __init__(self, learning_rate, iterations, l2_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penality = l2_penality
        
    def fit(self, X, Y):
        #dimension
        self.m, self.n = X.shape
        
        #initialization
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        
        #gradient descent
        for i in range(self.iterations):
            
            Y_pred = self.X.dot(self.w)+self.b
            error = Y_pred-self.Y #nx1
            
            dw = (self.X.T.dot(error)+self.l2_penality*self.w)/self.m #px1
            db = np.sum(error)/self.m #1x1
            self.w = self.w-self.learning_rate*dw
            self.b = self.b-self.learning_rate*db
        
        return
    
    def predict( self, X ) :     
        return X.dot( self.w ) + self.b
    
#train and test
def main():
    data = pd.read_excel('energy_data.xlsx')
    X = data.iloc[:, :-1].values 
    Y = data.iloc[:, 1].values
    
    #standardize data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                        test_size = 0.2, random_state = 0 )     
    # Model training     
    model = Ridge_Regression( iterations = 1000,                              
                            learning_rate = 0.01, l2_penality = 1 ) 
    model.fit( X_train, Y_train )
    
    #Model evaluation
    Y_pred = model.predict(X_test)
    
    #r2
    sst = np.sum((Y_test-Y_test.mean())**2)
    ssr = np.sum((Y_pred-Y_test)**2)
    r2 = 1-(ssr/sst)
    print('R2 score:',r2)
   
    
if __name__ == "__main__" :  
    main() 
        