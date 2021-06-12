# MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems

This is our PyTorch implementation for the paper:

> Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, Jie Tang (2021). MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems.  [Paper link](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf). In KDD'2021, Virtual Event, Singapore, August 14-18, 2021.

Author: Mr. Tinglin Huang (tinglin.huang at zju.edu.cn)

## Citation 

If you want to use our codes in your research, please cite:
​    
```
@inproceedings{MixGCF2021,
  author    = {Tinglin Huang and
               Yuxiao Dong and
               Ming Ding and
               Zhen Yang and
               Wenzheng Feng and
               Xinyu Wang and
               Jie Tang},
  title     = {MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems},
  booktitle = {{KDD}},
  year      = {2021}
}
```

## Environment Requirement

The code has been tested running under Python 3.7.6. The required packages are as follows:

- pytorch == 1.7.0
- numpy == 1.20.2
- scipy == 1.6.3
- sklearn == 0.24.1
- prettytable == 2.1.0

## Training

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). Important argument:

- `K`
  - It specifies the number of negative instances in K-pair loss. Note that when K=1 (by default), the K-pair loss will degenerate into the BPR pairwise loss.
- `n_negs`
  - It specifies the size of negative candidate set when using MixGCF.
- `ns`
  - It indicates the type of negative sample method. Here we provide two options: rns and mixgcf.

#### LightGCN

##### Random sample(rns)

```
python main.py --dataset ali --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --agg mean --ns rns --K 1 --n_negs 1

python main.py --dataset yelp2018 --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --agg mean --ns rns --K 1 --n_negs 1

python main.py --dataset amazon --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --agg mean --ns rns --K 1 --n_negs 1
```

#####  MixGCF

```
python main.py --dataset ali --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --agg mean --ns mixgcf --K 1 --n_negs 32

python main.py --dataset yelp2018 --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --agg mean --ns mixgcf --K 1 --n_negs 64

python main.py --dataset amazon --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --agg mean --ns mixgcf --K 1 --n_negs 16
```

#### NGCF

##### Random sample(rns)

```
python main.py --dataset ali --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0 --context_hops 3 --agg concat --ns rns --K 1 --n_negs 1

python main.py --dataset yelp2018 --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0 --context_hops 3 --agg concat --ns rns --K 1 --n_negs 1

python main.py --dataset amazon --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0 --context_hops 3 --agg concat --ns rns --K 1 --n_negs 1
```

##### MixGCF

```
python main.py --dataset ali --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0 --context_hops 3 --agg concat --ns mixgcf --K 1 --n_negs 64

python main.py --dataset yelp2018 --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0 --context_hops 3 --agg concat --ns mixgcf --K 1 --n_negs 64

python main.py --dataset amazon --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0 --context_hops 3 --agg concat --ns mixgcf --K 1 --n_negs 64
```

The [training log](https://github.com/huangtinglin/MixGCF/tree/main/training_log) is also provided. The results fluctuate slightly under different running environment.

## Dataset

We use three processed datasets: Alibaba, Yelp2018, and Amazon.

|               | Alibaba | Yelp2018  | Amazon    |
| ------------- | ------- | --------- | --------- |
| #Users        | 106,042 | 31,668    | 192,403   |
| #Items        | 53,591  | 38,048    | 63,001    |
| #Interactions | 907,407 | 1,561,406 | 1,689,188 |
| Density       | 0.00016 | 0.00130   | 0.00014   |

