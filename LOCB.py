
from collections import defaultdict
import numpy as np
import time
import random
from scipy.stats import ortho_group
from Environment import Environment
from utlis import generate_items, edge_probability
import sys 




class Base:
    # Base agent for LOCB
    def __init__(self, d, T):
        self.d = d
        self.T = T
        self.rewards = np.zeros(self.T)
        self.best_rewards = np.zeros(self.T)

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))

    def recommend(self, i, items, t):
        # items is of type np.array (L, d)
        # select one index from items to user i
        return

    def store_info(self, i, x, y, t, r, br):
        return

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta

    def update(self, i, t):
        return
        
    def update_LOCB(self, i, t):
        return

    def run(self, envir):
        for t in range(1, self.T):
            i = t%envir.nu
            items = envir.get_items()
            kk = self.recommend(i=i, items=items, t=t)
            x = items[kk]
            y, r, br = envir.feedback(i=i, k=kk)
            self.store_info(i=i, x=x, y=y, t=t, r=r, br=br)
            self.update_LOCB(i, t) 
            self.update(i, t)



class Cluster:
    def __init__(self, users, S, b, N):
        self.users = set(users) # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)


class LOCB(Base):
    # each user is an independent LinUCB
    def __init__(self, nu, d, T, gamma, num_seeds):
        super(LOCB, self).__init__(d, T)
        self.S = {i:np.eye(d) for i in range(nu)}
        self.b = {i:np.zeros(d) for i in range(nu)}
        self.Sinv = {i:np.eye(d) for i in range(nu)}
        self.theta = {i:np.zeros(d) for i in range(nu)}
        self.users = range(nu)
        self.seeds = np.random.choice(self.users, num_seeds)
        self.seed_state = {}
        for seed in self.seeds:
            self.seed_state[seed] = 0
            
        print('seed', self.seeds)
        self.clusters = {}
        for seed in self.seeds: 
            self.clusters[seed] = Cluster(users=self.users, S=np.eye(d), b=np.zeros(d), N=1)
        self.N = np.zeros(nu)
        self.gamma = gamma
        self.results = []
        self.fin = 0
        self.cluster_inds = {i:[] for i in range(nu)}
        for i in self.users:
            for seed in self.seeds:
                if i in self.clusters[seed].users:
                    self.cluster_inds[i].append(seed)
                    
        self.d = d
        self.n = nu
        self.selected_cluster = 0 
        self.delta = 0.1

    def recommend(self, i, items, t):
        cls = self.cluster_inds[i]
        if len(cls)>3:
            res = []
            for c in cls:
                cluster = self.clusters[c]
                res_sin = self._select_item_ucb(cluster.S,cluster.Sinv, cluster.theta, items, cluster.N, t)
                res.append(res_sin)
            best_cluster = max(res)
            if t%20000 ==0:
                print('best:',best_cluster)
                print('current clusters:', cls)
            return best_cluster[1]
        else:
            #print(best_cluster)
            no_cluster = self._select_item_ucb(self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t)
            return no_cluster[1]
  
        
    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        ucbs = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1)
        res = max(ucbs)
        it = np.argmax(ucbs)
        return (res, it)
        

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br
        
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])
        
        for c in self.cluster_inds[i]:
            self.clusters[c].S += np.outer(x, x)
            self.clusters[c].b += y * x
            self.clusters[c].N += 1
            
            self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
            self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)
            

        
    
    def update_LOCB(self, i, t):
        def _factT(T):
            delta = 6*self.delta / (self.n * np.pi * np.pi * t * t)
            nu = np.sqrt(2*self.d*np.log(1 + t) + 2*np.log(2/delta)) +1
            de = np.sqrt(1+T/8)*np.power(self.n, 1/3)
            return nu/de
        def _factT(m):
            return np.sqrt((1 + np.log(1 + m)) / (1 + m))
        
        if not self.fin:
    
            if t%10000 ==0:
                print('seed bound', _factT(self.N[0]), 'round:', t)
                #for seed in self.seeds:
                print(self.cluster_inds[i])
              
                

            for seed in self.seeds:
                if not self.seed_state[seed]:
                    if i in self.clusters[seed].users:
                        diff = self.theta[i] - self.theta[seed]
                        if np.linalg.norm(diff) > _factT(self.N[i]) + _factT(self.N[seed]):
                            self.clusters[seed].users.remove(i)
                            self.cluster_inds[i].remove(seed)                            
                            self.clusters[seed].S = self.clusters[seed].S - self.S[i] + np.eye(self.d)
                            self.clusters[seed].b = self.clusters[seed].b - self.b[i]
                            self.clusters[seed].N = self.clusters[seed].N - self.N[i]
                    else:
                        diff = self.theta[i] - self.theta[seed]
                        if np.linalg.norm(diff) < _factT(self.N[i]) + _factT(self.N[seed]):
                            self.clusters[seed].users.add(i)
                            self.cluster_inds[i].append(seed)
                            self.clusters[seed].S = self.clusters[seed].S + self.S[i] - np.eye(self.d)
                            self.clusters[seed].b = self.clusters[seed].b + self.b[i]
                            self.clusters[seed].N = self.clusters[seed].N + self.N[i]
          
                         
                    if _factT(self.N[seed]) <= (self.gamma/4):
                        self.seed_state[seed] = 1
                        self.results.append({seed:list(self.clusters[seed].users)}) 

            finished = 1
            for i in self.seed_state.values():
                if i ==0:
                    finished =0
            if finished: 
                self.fin = 1
                #print(" finished clustering")

