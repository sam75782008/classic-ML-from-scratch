#ML from scratch
#Naive Bayes Classifier

#packages
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 

#NB
class NB():
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        return
    
    def likelihood(self, X, mean, std):
        #Gaussian distribution
        #calculate mean and standard deviation for likelihood
        X = np.array(X)
        mean = np.array(mean)
        std = np.array(std)
        return np.exp(-(X-mean)**2/(2*std**2))*(1/(np.sqrt(2*np.pi)*std))
    
    def prior(self):
        m = self.Y.shape[0]
        num_class = len(set(self.Y))
        p_c = np.zeros((num_class,1))
        for i in range(num_class):
            p_c[i] = sum(self.Y==i)/m
        return p_c
    
    def posterior(self,likelihood, prior):
        product=np.prod(likelihood,axis=1)
        product=product*prior
        return product
    
    def fit(self):
        #claculate prior
        self.p_c = self.prior()
        
        #split class
        self.class_dict = dict()
        for i in range(self.X.shape[0]):
            class_name = self.Y[i]
            
            if class_name not in self.class_dict:
                self.class_dict[class_name] = []
            
            self.class_dict[class_name].append(self.X[i,:])
        
        #clacluate likelihood
        self.feature_mean = dict()
        self.feature_std = dict()
        
        for class_name in set(self.Y):
            
            if class_name not in self.feature_mean:
                self.feature_mean[class_name] = []
            
            if class_name not in self.feature_std:
                self.feature_std[class_name] = []
            
            self.feature_mean[class_name].append(np.mean(self.class_dict[class_name],axis=0))
            self.feature_std[class_name].append(np.std(self.class_dict[class_name],axis=0))

        return
    
    def predict(self,X_test):
        y_pred = []
        for example in range(X_test.shape[0]):
            x = X_test[example,:]
            temp = []
            for i in range(len(set(self.Y))):
                prior = self.p_c[i]
                likelihood = self.likelihood(x,self.feature_mean[i],self.feature_std[i])
                posterior = self.posterior(likelihood, prior)
                temp.append(posterior)
            y_pred.append(temp.index(max(temp)))
        
        return y_pred
        


def main():
    #dataset
    iris = datasets.load_iris()
    X = iris.data[:, :10]
    Y = (iris.target) * 1
       
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                                        test_size = 0.2, random_state = 0 )  
    # Model training     
    model = NB(X_train,Y_train)
    model.fit()

    #Model evaluation
    Y_pred = model.predict(X_test)
    print ('Accuracy from scratch:',(Y_pred == Y_test).sum().astype(float) / len(Y_pred))  

    
if __name__=='__main__':
    main()
