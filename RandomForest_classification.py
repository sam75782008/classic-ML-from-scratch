#Random Forest Classification Tree
import numpy as np

#load dataset
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
#Random Forest
class RandomForest():
    def __init__(self, x, y, n_trees, sample_sz=None, min_leaf=5):
        np.random.seed(12)
        
        if sample_sz is None:
            sample_sz=len(y)
        self.x, self.y, self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]
        
    
    def create_tree(self):
        idxs = np.random.choice(len(self.y), replace=True, size = self.sample_sz)
        return DecisionTree(self.x[idxs],self.y[idxs],
                        idxs=np.array(range(self.sample_sz)), min_leaf=self.min_leaf)
    
    def predict(self,x):
        percents = np.mean([t.predict(x) for t in self.trees],axis=0)
        return [1 if p>0.5 else 0 for p in percents]

#calculate gini score at given feature and given observation
def find_gini(left, right, y):
    classes = np.unique(y)
    n = len(left) + len(right)
    s1 = 0; s2 = 0
    
    for k in classes:
        p1 = len(np.nonzero(y[left] == k)[0])/len(left) #p(y|left)
        s1 += p1*p1
        p2 = len(np.nonzero(y[right] == k)[0])/len(right) #p(y|right)
        s2 += p2*p2
        
    gini = (1-s1)*(len(left)/n) + (1-s2)*(len(right)/n)
    
    return gini       

#Decision Tree
class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=5):
        if idxs is None:
            idxs = np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score=float('inf')
        self.check_features()
    
    
    def check_features(self): #recursively find the best split
        for i in range(self.c):
            self.find_best_split(i)
        if self.is_leaf: return
        
        #otherwise this split becones the root for a new tree
        x = self.split_cols
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])
   
    
    #find the best feature and standard based on gini
    def find_best_split(self, var_idx):
        x, y = self.x[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]
        
        for i in range(0, self.n-self.min_leaf+1):
            if i <self.min_leaf or sort_x[i] == sort_x[i+1]: continue
            lhs = np.nonzero(sort_x<=sort_x[i])[0]
            rhs = np.nonzero(sort_x>sort_x[i])[0]
            if rhs.sum()==0: continue
        
            gini = find_gini(lhs, rhs, sort_y)
            
            if gini<self.score:
                self.var_idx, self.score, self.split = var_idx, gini, sort_x[i]
    
    
    @property
    def split_name(self): return self.x[self.var_idx]
    
    @property
    def split_cols(self): return self.x[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self): return self.score == float('inf')
    
    
    #prediction
    def predict(self,x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self,xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)
        
        
