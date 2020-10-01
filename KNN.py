# ML from scratch
# K nearest neighbor

#ｉｍｐｏｒｔ　ｐａｃｋａｇｅ
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN():
    def __init__(self,neighbor):
        self.neighbor = neighbor
    
    def euclidean(self, X1, X2):
        diff = X1 - X2
        return np.sqrt(np.dot(diff,diff.T))
    
    def predict(self, X_train, X_test, Y_train):
        Y_pred = []
        #calculate distance between each data point and the test data 
        for test_row in range(len(X_test)):
            distance = []
            for train_row in range(len(X_train)):
                distance.append((self.euclidean(X_train[train_row,:],X_test[test_row,:]),Y_train[train_row]))
                
            #sort distance
            distance.sort(key=lambda x: x[0])

            #count catagory
            candidate = [out[-1] for out in distance[:self.neighbor]]

            
            #predict
            prediction = max(set(candidate), key=candidate.count)
            Y_pred.append(prediction)
        
        return Y_pred

def main():
    #dataset
    iris = datasets.load_iris()
    X = iris.data[:, :10]
    Y = (iris.target) * 1
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                        test_size = 0.2, random_state = 0 )     
    # Model   
    model = KNN(neighbor=5) 
    Y_pred = model.predict(X_train, X_test, Y_train)
    print ('Accuracy from scratch:',(Y_pred == Y_test).sum().astype(float) / len(Y_pred))
    
if __name__ == "__main__" :  
    main() 
            
            
            
            
                
                
        