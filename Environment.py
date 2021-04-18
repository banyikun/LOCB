import numpy as np
import sys

def get_best_reward(items, theta):
    return np.max(np.dot(items, theta))

def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon

def edge_probability(n):
    return 3 * np.log(n) / n

def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)

def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d-1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((num_items, 1))/np.sqrt(2)), axis = 1)
    return x

def evaluate_clustering(groups, groups_l):
    b_f1s = []
    b_pres = []
    b_recall = []
    for i in groups_l:
        f1s = 0.0
        pres = 0.0
        recalls = 0.0
        for j in groups:
            right = 0.0
            for u in j:
                if u in i:
                    right +=1
            pre = right/len(j)
            recall = right/len(i)
            f1 = 0.0
            if pre+recall > 0:
                f1 = 2*pre*recall/(pre+recall)
            if f1 > f1s:
                f1s = f1
                pres = pre
                recalls = recall
        b_f1s.append(f1s)
        b_pres.append(pres)
        b_recall.append(recalls)
    return np.average(b_f1s), np.average(b_pres), np.average(b_recall)



class Environment:
    # p: frequency vector of users
    def __init__(self, L, d, m, num_users, p, theta):
        self.L = L
        self.d = d
        self.p = p # probability distribution over users

        self.items = generate_items(num_items = L, d = d)
        self.theta = theta
        self.nu = num_users

    def get_items(self):
        self.items = generate_items(num_items = self.L, d = self.d)
        return self.items

    def feedback(self, i, k):
        x = self.items[k, :]
        r = np.dot(self.theta[i], x)
        y = np.random.binomial(1, r)
        br = get_best_reward(self.items, self.theta[i])
        return y, r, br

    def generate_users(self):
        X = np.random.multinomial(1, self.p)
        I = np.nonzero(X)[0]
        return I
    