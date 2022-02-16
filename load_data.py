from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 

import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from collections import defaultdict

        
        
class load_notmnist_mnist_2:
    def __init__(self):
        
        # Not mnist
        X = np.load('./not_mnist_data.npy', allow_pickle=True)
        y = np.load('./not_mnist_label.npy', allow_pickle=True)
        new_X = []
        for i in X:
            i = i.flatten()
            new_X.append(i)
        X = np.array(new_X)
        X[np.isnan(X)] = - 1
        X = normalize(X)
        self.X, self.y =shuffle(X, y)
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] + 9
        self.act_dim = self.X.shape[1]
        
        #  mnist
        batch_size = 1
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                      shuffle=True, num_workers=2)
        self.dataiter = iter(train_loader)
        self.num_user = 20

    def step(self):
        
        choice = np.random.choice(2, 1, replace=True)[0]
        if choice == 0:
            assert self.cursor < self.size
            X = np.zeros((self.n_arm, self.dim))
            for a in range(self.n_arm):
                X[a, a:a+
                    self.act_dim] = self.X[self.cursor]
            arm = self.y_arm[self.cursor][0]
            rwd = np.zeros((self.n_arm,))
            rwd[arm] = 1
            self.cursor += 1
            return arm,  X, rwd
        else:
            x, y = self.dataiter.next()
            d = x.numpy()[0]
            d = d.reshape(self.act_dim )
            target = y.item()
            X = np.zeros((self.n_arm, self.dim))
            for a in range(self.n_arm):
                X[a, a:a+
                    self.act_dim] = d
            rwd = np.zeros(self.n_arm)
            #print(target)
            rwd[target] = 1
            return 10+target, X, rwd  

        
    

    

    
    
      
    
    
    

    
    
from scipy.stats import norm
class load_movielen_new:
    def __init__(self):
        # Fetch data
        self.m = np.load("./movie_2000users_10000items_entry.npy")
        self.U = np.load("./movie_2000users_10000items_features.npy")
        self.I = np.load("./movie_10000items_2000users_features.npy")
        kmeans = KMeans(n_clusters=50, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 10
        self.num_user = 50
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in self.m:
            if i[2] ==1:
                self.pos_index[self.groups[i[0]]].append((i[0], i[1]))
            else:
                self.neg_index[self.groups[i[0]]].append((i[0], i[1]))   


    def step(self):    
        u = np.random.choice(range(2000))
        g = self.groups[u]
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[ind[1]])
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        return g, contexts, rwd
    
    
class load_yelp_new:
    def __init__(self):
        # Fetch data
        self.m = np.load("./yelp_2000users_10000items_entry.npy")
        self.U = np.load("./yelp_2000users_10000items_features.npy")
        self.I = np.load("./yelp_10000items_2000users_features.npy")
        kmeans = KMeans(n_clusters=50, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 10
        self.num_user = 50
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in self.m:
            if i[2] ==1:
                self.pos_index[self.groups[i[0]]].append((i[0], i[1]))
            else:
                self.neg_index[self.groups[i[0]]].append((i[0], i[1]))   
   

    def step(self):    
        u = np.random.choice(range(2000))
        g = self.groups[u]
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[ind[1]])
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        return g, contexts, rwd

        
        
        