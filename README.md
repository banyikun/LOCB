## Local Clustering in Contextual Multi-Armed Bandits

### Abstract

We study identifying user clusters in contextual multi-armed bandits (MAB). Contextual MAB is an effective tool for many real applications, such as content recommendation and online adver- tisement. In practice, user dependency plays an essential role in the user’s actions, and thus the rewards. Clustering similar users can improve the quality of reward estimation, which in turn leads to more effective content recommendation and targeted advertising. Different from traditional clustering settings, we cluster users based on the unknown bandit parameters, which will be estimated incre- mentally. In particular, we define the problem of cluster detection in contextual MAB, and propose a bandit algorithm, LOCB, embed- ded with local clustering procedure. And, we provide theoretical analysis about LOCB in terms of the correctness and efficiency of clustering and its regret bound. Finally, we evaluate the proposed algorithm from various aspects, which outperforms state-of-the-art baselines.


### Requirements

Python 3.7
numpy
networkx

### Command

#### Default:

For regret analysis:
python3 main.py --num_stages 15 --num_users 100 --num_items 20 --d 5 --m 5 --gamma 0.2 --delta 0.1 --num_seeds 60 --detect_cluster 0

Reference:
