import numpy as np
from LOCB import LOCB
from Environment import Environment


def main(num_stages, num_users, d, m, L, pj, filename=''):


    def generate_items(num_items, d):
        # return a ndarray of num_items * d
        x = np.random.normal(0, 1, (num_items, d-1))
        x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((num_items, 1))/np.sqrt(2)), axis = 1)
        return x

            # set up theta vector
    def _get_theta(thetam, num_users, m):
        k = int(num_users / m)
        theta = {i:thetam[0] for i in range(k)}
        labels = [0 for i in range(num_users)]
        for j in range(1, m):
            for i in range(k * j, k * (j + 1)):
                labels[i] =j
                theta.update({i:thetam[j]})
        return theta, labels
        
        
    if filename == 'synthetic':
        thetam = generate_items(num_items=m, d=d)
        theta, labels = _get_theta(thetam, num_users, m)


    envir = Environment(L = L, d = d, m = m, num_users = num_users, p = p, theta = theta)
        
    locb = LOCB(nu = num_users, d = d, T = 2 ** num_stages - 1, gamma = 0.2, num_seeds = 30)
    start_time = time.time()
    locb.run(envir)
    run_time = time.time() - start_time
    np.save('./results/%s/regret_%s'%(dataset,num_stages), np.subtract(locb.best_rewards, locb.rewards))
    np.save('./results/%s/reward_%s'%(dataset,num_stages), locb.rewards)
    
        
  
if __name__== "__main__":
    main(num_stages = 14, num_users = 200, d = 5, m = 5, L = 20, pj = 0, filename='')