import os
import argparse
from LOCB import LOCB
from Environment import Environment
from Environment import generate_items
from Environment import evaluate_clustering
import numpy as np
from collections import defaultdict



def main(num_stages, num_users, num_items, d, m, gamma, num_seeds, delta, detect_cluster,  dataset='sync'):
                
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
        
        
    thetam = generate_items(num_items=m, d=d)
    theta, labels = _get_theta(thetam, num_users, m)
    
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
    if not os.path.exists('./results'):
        os.makedirs('./results')
        
    np.save('./datasets/syn_labels', labels)
    np.save('./datasets/syn_data', theta)

    def generate_user_probability(num_users, m):
        p = np.ones(num_users)
        for i in range(num_users):
            p[i] = 1.0 / num_users
        p = list(p)
        return p
    
    p = generate_user_probability(num_users=num_users,m=m)
    envir = Environment(L = num_items, d = d, m = m, num_users = num_users, p = p, theta = theta)

    locb = LOCB(nu = num_users, d = d, T = 2 ** num_stages - 1, gamma = gamma, num_seeds = num_seeds, delta = delta,  detect_cluster = detect_cluster)
    locb.run(envir)
    np.save('./results/regret.npy', np.subtract(locb.best_rewards, locb.rewards))
    np.save('./results/reward.npy', locb.rewards)
    
    if detect_cluster==0:
        print('glub_ind_regret:', np.sum(np.subtract(glub.best_rewards, glub.rewards)))
    else:
        g_t = np.load('./datasets/syn_labels.npy')        
        dic_u = defaultdict(lambda : []) 
        for i in range(len(g_t)):
            dic_u[g_t[i]].append(i)
        groups_l = list(dic_u.values())

        res = np.load('./results/clusters.npy', allow_pickle = True)
        groups = []
        for i in res:
            g = list(i.values())[0]
            groups.append(g)   

        f1, pre, rec = evaluate_clustering(groups, groups_l)
        print('Clustering. F1:', f1, 'Precision:', pre, 'Recall:', rec)
       


if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description='LOCB')

    parser.add_argument('--num_stages', default=15, type=int, help = 'number of rounds: 2**num_stages')
    parser.add_argument('--num_users', default=100, type=int, help = 'number of users')
    parser.add_argument('--num_items', default=20, type=int, help = 'number of items')
    parser.add_argument('--d', default=5, type=int, help = 'number of dimensions for user and item features')
    parser.add_argument('--m', default=5, type=int, help = 'number of clusters in synthetic data')
    parser.add_argument('--gamma', default=0.2, type=float, help='parameter in LOCB')
    parser.add_argument('--delta', default=0.1, type=float, help='confidence interval')
    parser.add_argument('--num_seeds', default=60, type=int, help='the number of seeds for LOCB')
    parser.add_argument('--detect_cluster', default=0, type=int, help='o: output regret; 1: output clustering accuracy')
    
    args = parser.parse_args()
    main(num_stages = args.num_stages, num_users = args.num_users, num_items=args.num_items, d = args.d, m = args.m, gamma = args.gamma, delta = args.delta, num_seeds = args.num_seeds, detect_cluster = args.detect_cluster)
    #main(num_stages = 15, num_users = 100, num_items=20,  d = 5, m = 5, gamma = 0.2, delta = 0.1, num_seeds = 60, detect_cluster = 0)



