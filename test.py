# coding: utf
from   __future__ import print_function

from   PIL import Image
import tensorflow as tf
import os
import glob
import csv
from   models import *
from   dataset_mini import *
from   dataset_tiered import *
from   tqdm import tqdm
import argparse
import random
import numpy as np
import scipy as sp
import scipy.stats


parser = argparse.ArgumentParser(description='Test TPN')

# parse gpu and random params
default_gpu = "0"
parser.add_argument('--gpu',            type=str,       default=0,          metavar='GPU',
                    help="gpu name, default:{}".format(default_gpu))
parser.add_argument('--seed',           type=int,       default=1000,       metavar='SEED',
                    help="random seed, -1 means no seed")

# model params
n_examples = 600
parser.add_argument('--x_dim',          type=str,       default="84,84,3",  metavar='XDIM',
                    help='input image dims')
parser.add_argument('--h_dim',          type=int,       default=64,         metavar='HDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--z_dim',          type=int,       default=64,         metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")

# basic training hyper-parameters
n_episodes = 100
parser.add_argument('--n_way',          type=int,       default=5,          metavar='NWAY',
                    help="nway")
parser.add_argument('--n_shot',         type=int,       default=5,          metavar='NSHOT',
                    help="nshot")
parser.add_argument('--n_query',        type=int,       default=15,         metavar='NQUERY',
                    help="nquery")
parser.add_argument('--n_epochs',       type=int,       default=1100,       metavar='NEPOCHS',
                    help="nepochs")

# val and test hyper-parameters
parser.add_argument('--n_test_way',     type=int,       default=5,          metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--n_test_shot',    type=int,       default=5,          metavar='NTESTSHOT',
                    help="ntestshot")
parser.add_argument('--n_test_query',   type=int,       default=15,         metavar='NTESTQUERY',
                    help="ntestquery")
parser.add_argument('--n_test_episodes',type=int,       default=600,        metavar='NTESTEPI',
                    help="ntestepisodes")

# optimization params
parser.add_argument('--lr',             type=float,     default=0.001,      metavar='LR',
                    help="base learning rate")
parser.add_argument('--step_size',      type=int,       default=2000,       metavar='DSTEP',
                    help="step size")
parser.add_argument('--gamma',          type=float,     default=0.5,        metavar='DRATE',
                    help="gamma")
parser.add_argument('--patience',       type=int,       default=200,        metavar='PATIENCE',
                    help="patience")

# dataset params
parser.add_argument('--dataset',        type=str,       default='mini',     metavar='DATASET',
                    help="mini or tiered")
parser.add_argument('--ratio',          type=float,     default=1.0,        metavar='RATIO',
                    help="ratio of labeled data")
parser.add_argument('--pkl',            type=int,       default=1,          metavar='PKL',
                    help="")

# label propagation params
parser.add_argument('--k',              type=int,       default=20,         metavar='K',
                    help="K in refine protos")
parser.add_argument('--sigma',          type=float,     default=0.25,       metavar='SIGMA',
                    help="SIGMA of k-NN graph construction")
parser.add_argument('--alpha',          type=float,     default=0.99,       metavar='ALPHA',
                    help="ALPHA in label propagation")
parser.add_argument('--rn',             type=int,       default=300,        metavar='RN',
                    help="relation types" + 
                         "30:learned sigma and alpha,    300:learned sigma, fixed alpha")

# restore params
parser.add_argument('--iters',          type=int,       default=0,          metavar='ITERS',
                    help="iteration to restore params")
parser.add_argument('--exp_name',       type=str,       default='exp',      metavar='EXPNAME',
                    help="experiment description name")


args = vars(parser.parse_args())
im_width, im_height, channels = list(map(int, args['x_dim'].split(',')))
print(args)
for key,v in args.items(): exec(key+'=v')


#seed = 1423
#random.seed(seed)
#np.random.seed(seed)
#tf.set_random_seed(seed)


os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
is_training = False



# construct dataset
if dataset=='mini':
    loader_test = dataset_mini(n_examples, n_episodes, 'test', args)
elif dataset=='tiered':
    loader_test = dataset_tiered(n_examples, n_episodes, 'test', args)

if not pkl:
    loader_test.load_data()
else:
    loader_test.load_data_pkl()


# construct model
m = models(args)
ce_loss,acc,sigma_value = m.construct()


# init session and start training
config   = tf.ConfigProto()  
config.gpu_options.allow_growth=True 
sess     = tf.Session(config=config)

init_op  = tf.global_variables_initializer()
sess.run(init_op)

saver    = tf.train.Saver(tf.global_variables(), max_to_keep=100)
save_dir = 'checkpoints/'+args['exp_name']

model_path = save_dir+'/models'
# restore pre-trained model
if iters>0:
    ckpt_path = model_path+'/ckpt-'+str(iters)
    
    saver.restore(sess, ckpt_path)
    print('Load model from {}'.format(ckpt_path))



print("Testing...")

all_acc  = []
all_std  = []
all_ci95 = []

nums     = n_test_way * (n_test_shot+n_test_query)


list_acc = []
# test epochs
for epi in tqdm(range(n_test_episodes), desc='test'):
    support, s_labels, query, q_labels, unlabel = loader_test.next_data(n_test_way, n_test_shot, n_test_query, train=False)
    vls, vac, vsigma = sess.run([ce_loss, acc, sigma_value], feed_dict={m.x: support, m.ys:s_labels, m.q: query, m.y:q_labels, m.phase:0})
    list_acc.append(vac)

mean_acc = np.mean(list_acc)
std_acc  = np.std(list_acc)
ci95     = 1.96*std_acc/np.sqrt(n_test_episodes)

print('Acc:{:.4f},std:{:.4f},ci95:{:.4f}'.format(mean_acc, std_acc, ci95))



