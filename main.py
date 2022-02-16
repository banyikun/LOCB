import os
import argparse
from LOCB import LOCB
import numpy as np
from collections import defaultdict
import sys 
from load_data import load_movielen_new, load_yelp_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-Ban')
    parser.add_argument('--dataset', default='movie', type=str, help='yelp, movie')
    args = parser.parse_args()
    data = args.dataset
    
    if data == "yelp":
        b = load_yelp_new()
        
    elif data == "movie":
        b = load_movielen_new()
    else:
        print("dataset is not defined. --help")
        sys.exit()
    
    

    model = LOCB(nu = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)
        
    regrets = []
    summ = 0
    print("Round; Regret; Regret/Round")
    for t in range(10000):
        u, context, rwd = b.step()
        arm_select = model.recommend(u, context, t)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)
        model.store_info(i = u, x = context[arm_select], y =r, t = t)
        model.update(i = u, t =t)           
        if t % 50 == 0:
            print('{}: {:}, {:.4f}'.format(t, summ, summ/(t+1)))
    print("round:", t, "; ", "regret:", summ)
    np.save("./regret",  regrets)
    
    
