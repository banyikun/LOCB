## Local Clustering in Contextual Multi-Armed Bandits

### Abstract

We study identifying user clusters in contextual multi-armed bandits (MAB). Contextual MAB is an effective tool for many real applications, such as content recommendation and online adver- tisement. In practice, user dependency plays an essential role in the user’s actions, and thus the rewards. Clustering similar users can improve the quality of reward estimation, which in turn leads to more effective content recommendation and targeted advertising. Different from traditional clustering settings, we cluster users based on the unknown bandit parameters, which will be estimated incre- mentally. In particular, we define the problem of cluster detection in contextual MAB, and propose a bandit algorithm, LOCB, embed- ded with local clustering procedure. And, we provide theoretical analysis about LOCB in terms of the correctness and efficiency of clustering and its regret bound. Finally, we evaluate the proposed algorithm from various aspects, which outperforms state-of-the-art baselines.


### Requirements

Python 3.7
numpy
networkx

### Command
The following commands report ther results on the synthetic dataset.

#### Default:

For regret analysis:
python3 main.py --num_stages 15 --num_users 100 --num_items 20 --d 5 --m 5 --gamma 0.2 --delta 0.1 --num_seeds 60 --detect_cluster 0

For clustering accuracy:
python3 main.py --num_stages 16 --num_users 100 --num_items 20 --d 5 --m 5 --gamma 0.27 --delta 0.1 --num_seeds 60 --detect_cluster 1


### Parameters

num_stages: number of rounds computed by 2 ** num_stages

num_users: number of users

num_items: number of items

d: number of dimensions for user and item features

m: number of clusters in synthetic data

gamma: parameter in LOCB

delta: confidence interval

num_seeds: the number of seeds for LOCB

detect_cluster: '0' for the regret comparison and output regret; '1' for detecting clusters and output clustering accuracy

Reference:
