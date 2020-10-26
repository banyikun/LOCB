import numpy as np

def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d-1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((num_items, 1))/np.sqrt(2)), axis = 1)
    return x


def get_best_reward(items, theta):
    return np.max(np.dot(items, theta))


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