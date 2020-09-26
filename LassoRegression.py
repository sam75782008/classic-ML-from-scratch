#ML from scratch
#Lasso Regression

#import package
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class Lasso_Regression():
    
    def __init__(self, iterations, l1_penality):
        self.iterations = iterations
        self.l1_penality = l1_penality
        
    #soft threshold
    def soft_threshold(self, rho, lamda, zk):
        if rho < -lamda/2:
            return (rho+lamda/2)/zk
        elif rho > lamda/2:
            return (rho-lamda/2)/zk
        else: 
            return 0
    
    def fit(self, X, Y):
        #dimension
        self.m, self.n = X.shape
        
        #initialization
        self.w = np.zeros(self.n)
        self.X = X
        self.Y = Y
        #coordinate descent
        for i in range(self.iterations):
            for j in range(self.n): #coordinate descent for each w
                zk = self.X[:,j].T.dot(self.X[:,j])
                rho = 0
                for k in range(self.m):
                    rho += self.X[k, j]*(self.Y[k] - np.sum([self.X[k,l]*self.w[l] for l in range(self.n) if l != j]))
                self.w[j] = self.soft_threshold(rho, self.l1_penality,zk)
        print(self.w)
        return
    
    def predict( self, X ) :     
        return X.dot( self.w )
    
#train and test
def main():
    data = pd.read_excel('energy_data.xlsx')
    X = data.iloc[:, :-1].values
    X = np.c_[np.ones(len(X),dtype='int64'),X]
    Y = data.iloc[:, 1].values
    
    #standardize data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                        test_size = 0.2, random_state = 0 )     
    # Model training     
    model = Lasso_Regression(iterations=100, l1_penality = 10 ) 
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
        