# Transductive Propagation Network
Code for ICLR19 paper: 
*Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning.* [pdf](https://openreview.net/pdf?id=SyVuRiC5K7)

## Pytorch Version 
https://github.com/csyanbin/TPN-pytorch

## Requirements
* Python 3.5
* Tensorflow 1.3+
* tqdm


## Data Download (miniImagenet and tieredImagenet)
Please download the compressed tar files from: https://github.com/renmengye/few-shot-ssl-public

```
mkdir -p data/miniImagenet/data
tar -zxvf mini-imagenet.tar.gz
mv *.pkl data/miniImagenet/data

mkdir -p data/tieredImagenet/data
tar -xvf tiered-imagenet.tar
mv *.pkl data/tieredImagenet/data

```

## TPN mini-5way1shot
```
python train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w1s_5tw1ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```

```
python test.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w1s_5tw1ts_rn300_k20 --rn=300 --alpha=0.99 --k=20 --iters=81500
```

## TPN mini-5way5shot
```
python train.py --gpu=0 --n_way=5 --n_shot=5 --n_test_way=5 --n_test_shot=5 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w5s_5tw5ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```

```
python test.py --gpu=0 --n_way=5 --n_shot=5 --n_test_way=5 --n_test_shot=5 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w5s_5tw5ts_rn300_k20 --rn=300 --alpha=0.99 --k=20 --iters=50100

```

## TPN tiered-5way1shot
```
python train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --step_size=25000 --dataset=tiered --exp_name=tiered_TPN_5w1s_5tw1ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```

## TPN tiered-5way5shot
```
python train.py --gpu=0 --n_way=5 --n_shot=5 --n_test_way=5 --n_test_shot=5 --lr=0.001 --step_size=25000 --dataset=tiered --exp_name=tiered_TPN_5w5s_5tw5ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```


## Citation
If you use our code, please consider to cite the following paper:
* Yanbin Liu, Juho Lee, Minseop Park, Saehoon Kim, Eunho Yang, Sungju Hwang, Yi Yang. Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning. In *Proceedings of 7th International Conference on Learning Representations (ICLR)*, 2019.

```

@inproceedings{liu2019fewTPN,
	title={Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning},
	author={Yanbin Liu and 
		Juho Lee and 
		Minseop Park and 
		Saehoon Kim and 
		Eunho Yang and 
		Sungju Hwang and 
		Yi Yang},
booktitle={International Conference on Learning Representations},
year={2019},
}

```

