# Random Forest from scratch
# This code practice to develop Random Forest regression model from scratch
import numpy as np
import math

from sklearn.datasets import make_regression
# load data set
X, Y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

#X: predictors
#Y: responses
#n_trees: number of trees to train
#n_features: number of features to be use in each tree
#sample_size: number of observations to be used
#depth: number of split
#min_leaf: mimimal number of observation to be split

class RandomForest():
    def __init__(self, x, y, n_trees, n_features, sample_sz, depth=10, min_leaf=5):
        np.random.seed(12)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf = x, y, sample_sz, depth, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]
        
    
    def create_tree(self):
        #idxs: random select observation and their index
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        #f_idxs: random select features and their index
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        #call decision tree
        return DecisionTree(self.x[idxs], self.y[idxs], self.n_features, f_idxs,
                            idxs=np.array(range(self.sample_sz)), depth=self.depth, min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees],axis=0)

#calculate standard deviation for score    
def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt)-(s1/cnt)**2)

class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs, idxs, depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1] #n:rows; c:columns
        self.val = np.mean(y[idxs]) #values (score) at root
        self.score = float('inf') #current score, infinite at beginning
        self.find_varsplit() #find out which value in the feature as standard for split
    
    def find_varsplit(self): #recursive
        for i in self.f_idxs: self.find_better_split(i) #go through each columns (features)
        if self.is_leaf: return #reach leaf, then stop split
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0] #left hand side observation
        rhs = np.nonzero(x>self.split)[0] #right hand side observation
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features] #random select features for next split
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features] #random select features for next split
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth-1, min_leaf=self.min_leaf) #left tree
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth-1, min_leaf=self.min_leaf) #right tree
        
    def find_better_split(self, var_idx):
        x, y = self.x[self.idxs,var_idx], self.y[self.idxs] #n observations, one feature:var_idx
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        #number of observation; sum of value; sum of square value-->for score calculation
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.
    
        #go through each ovservation: greedy algorithm
        for i in range(0, self.n-self.min_leaf-1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1; rhs_cnt -=1 #left leaf and right leaf
            lhs_sum += yi; rhs_sum -=yi #left value and right value
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2 #left square value and right square value
            if i<self.min_leaf or xi==sort_x[i+1]:
                continue
            
            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2) #left leaf standard deviation
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2) #right leaf standard deviation
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt #weighted average score
            #record the mini score and the featues as well as the rows for current split
            if curr_score<self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi
    
    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self): return self.score == float('inf') or self.depth<=0
    
    
    def predict(self,x):
        return np.array([self.predict_row(xi) for xi in x]) #predict each rows
    
    def predict_row(self, xi): #recursively run through each leaf in tree
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)