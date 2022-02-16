from collections import defaultdict
import numpy as np
import random
import sys 
import networkx as 


class Cluster:
    def __init__(self, users, S, b, N):
        self.users = set(users) # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)


class LOCB:
    def __init__(self, nu, d, gamma, num_seeds, delta, detect_cluster):
        self.S = {i:np.eye(d) for i in range(nu)}
        self.b = {i:np.zeros(d) for i in range(nu)}
        self.Sinv = {i:np.eye(d) for i in range(nu)}
        self.theta = {i:np.zeros(d) for i in range(nu)}
        self.users = range(nu)
        
        self.seeds = np.random.choice(self.users, num_seeds)
        self.seed_state = {}
        for seed in self.seeds:
            self.seed_state[seed] = 0
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
        self.delta = delta
        self.if_d = detect_cluster
        
    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))


    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta
    

    def recommend(self, i, items, t):
        cls = self.cluster_inds[i]
        if (len(cls)>0) and (t <40000):
            res = []
            for c in cls:
                cluster = self.clusters[c]
                res_sin = self._select_item_ucb(cluster.S,cluster.Sinv, cluster.theta, items, cluster.N, t)
                res.append(res_sin)
            best_cluster = max(res)
            return best_cluster[1]
        else:
            no_cluster = self._select_item_ucb(self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t)
            return no_cluster[1]
  
        
    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        ucbs = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1)
        res = max(ucbs)
        it = np.argmax(ucbs)
        return (res, it)
        

    def store_info(self, i, x, y, t):
        
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
            

        
    
    def update(self, i, t):
        def _factT(m):
            if self.if_d:
                delta = self.delta / self.n
                nu = np.sqrt(2*self.d*np.log(1 + t) + 2*np.log(2/delta)) +1
                de = np.sqrt(1+m/4)*np.power(self.n, 1/3)
                return nu/de
            else:
                return np.sqrt((1 + np.log(1 + m)) / (1 + m))
        
        if not self.fin:

              
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
                
                    if self.if_d: thre = self.gamma 
                    else: thre = self.gamma/4
                        
                    if _factT(self.N[seed]) <= thre:
                        self.seed_state[seed] = 1
                        self.results.append({seed:list(self.clusters[seed].users)}) 

            finished = 1
            for i in self.seed_state.values():
                if i ==0:
                    finished =0
                    
            if finished: 
                if self.if_d:
                    np.save('./results/clusters', self.results)
                    print('Clustering finished! Round:', t)
                    self.stop = 1
                self.fin = 1

                    
                                
 
