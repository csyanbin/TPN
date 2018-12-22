#-------------------------------------
# Paper:  Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning
# Date:   2018.11.17
# Author: Anonymous
# All Rights Reserved
#-------------------------------------

from   __future__ import print_function
from   PIL import Image
import numpy as np
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

parser = argparse.ArgumentParser(description='Train TPN')

# parse gpu 
parser.add_argument('--gpu',        type=str,    default=0,     metavar='GPU',
                    help="gpu name, default:0")

# model params
n_examples = 600
parser.add_argument('--x_dim',      type=str,    default="84,84,3", metavar='XDIM',
                    help='input image dims')
parser.add_argument('--h_dim',      type=int,    default=64,    metavar='HDIM',
                    help="channels of hidden conv layers (default: 64)")
parser.add_argument('--z_dim',      type=int,    default=64,    metavar='ZDIM',
                    help="channels of last conv layer (default: 64)")

# training hyper-parameters
n_episodes = 100
parser.add_argument('--n_way',      type=int,    default=5,     metavar='NWAY',
                    help="nway")
parser.add_argument('--n_shot',     type=int,    default=5,     metavar='NSHOT',
                    help="nshot")
parser.add_argument('--n_query',    type=int,    default=15,    metavar='NQUERY',
                    help="nquery")
parser.add_argument('--n_epochs',   type=int,    default=2100,  metavar='NEPOCHS',
                    help="nepochs")
# test hyper-parameters
parser.add_argument('--n_test_way', type=int,    default=5,     metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--n_test_shot',type=int,    default=5,     metavar='NTESTSHOT',
                    help="ntestshot")
parser.add_argument('--n_test_query',type=int,   default=15,    metavar='NTESTQUERY',
                    help="ntestquery")

# optimization params
parser.add_argument('--lr',         type=float,  default=0.001, metavar='LR',
                    help="base learning rate")
parser.add_argument('--step_size',  type=int,    default=10000, metavar='DSTEP',
                    help="step_size")
parser.add_argument('--gamma',      type=float,  default=0.5,   metavar='DRATE',
                    help="gamma")
parser.add_argument('--patience',   type=int,    default=200,   metavar='PATIENCE',
                    help="patience")

# dataset params
parser.add_argument('--dataset',    type=str,    default='mini',metavar='DATASET',
                    help="mini or tiered")
parser.add_argument('--ratio',      type=float,  default=1.0,   metavar='RATIO',
                    help="ratio of labeled data (for semi-supervised setting")
parser.add_argument('--pkl',        type=int,    default=1,     metavar='PKL',
                    help="1 for use pkl dataset, 0 for original images")

# label propagation params
parser.add_argument('--k',          type=int,    default=20,    metavar='K',
                    help="top k in constructing the graph W")
parser.add_argument('--sigma',      type=float,  default=0.25,  metavar='SIGMA',
                    help="sigma of graph computing parameter")
parser.add_argument('--alpha',      type=float,  default=0.99,  metavar='ALPHA',
                    help="alpha in label propagation")
parser.add_argument('--rn',         type=int,    default=300,   metavar='RN',
                    help="graph construction types: "
                    "300: sigma is learned, alpha is fixed" +
                    "30:  both sigma and alpha learned")

# seed and exp_name
parser.add_argument('--seed',       type=int,    default=1000,  metavar='SEED',
                    help="random seed, -1 means no seed")
parser.add_argument('--exp_name',   type=str,    default='exp', metavar='EXPNAME',
                    help="experiment description name")
parser.add_argument('--iters',      type=int,    default=0,    metavar='ITERS',
                    help="checkpoint restore iters")



# deal with params
args = vars(parser.parse_args())
im_width, im_height, channels = list(map(int, args['x_dim'].split(',')))
for key,v in args.items(): exec(key+'=v')

## RANDOM SEED
#random.seed(seed)
#np.random.seed(seed)
#tf.set_random_seed(seed)

# set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
is_training = True


# deal with checkpoints save folder
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args['exp_name']):
        os.makedirs('checkpoints/'+args['exp_name'])
    if not os.path.exists('checkpoints/'+args['exp_name']+'/'+'models'):
        os.makedirs('checkpoints/'+args['exp_name']+'/'+'models')
    if not os.path.exists('checkpoints/'+args['exp_name']+'/'+'summaries'):
        os.makedirs('checkpoints/'+args['exp_name']+'/'+'summaries')
    os.system('cp train.py checkpoints'+'/'+args['exp_name']+'/'+'train.py.backup')
    os.system('cp models.py checkpoints' + '/' + args['exp_name'] + '/' + 'models.py.backup')
    f = open('checkpoints/'+args['exp_name']+'/log.txt', 'a')
    print(args, file=f)
    f.close()
_init_()


# construct dataset
if dataset=='mini':
    loader_train = dataset_mini(n_examples, n_episodes, 'train', args)
    loader_val   = dataset_mini(n_examples, n_episodes, 'val', args)
elif dataset=='tiered':
    loader_train = dataset_tiered(n_examples, n_episodes, 'train', args)
    loader_val   = dataset_tiered(n_examples, n_episodes, 'val', args)

if pkl==0:
    print('Load image data rather than PKL')
    loader_train.load_data()
    loader_val.load_data()
else:
    print('Load PKL data')
    loader_train.load_data_pkl()
    loader_val.load_data_pkl()


# construct model
m = models(args)
ce_loss,acc,sigma_value = m.construct()


# train and stepsize
global_step   = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step,
                                                step_size, gamma, staircase=True)
# update ops for batch norm
update_ops    = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op  = tf.train.AdamOptimizer(learning_rate).minimize(ce_loss, global_step=global_step)

# init session and start training
config  = tf.ConfigProto()  
config.gpu_options.allow_growth=True 
sess    = tf.Session(config=config)

init_op = tf.global_variables_initializer()
sess.run(init_op)


# summary
save_dir = 'checkpoints/'+args['exp_name']

loss_summary  = tf.summary.scalar("loss", ce_loss)
acc_summary   = tf.summary.scalar("accuracy", acc)
lr_summary    = tf.summary.scalar("lr", learning_rate)
sigma_summary = tf.summary.histogram("sigma", sigma_value)

train_summary_op     = tf.summary.merge([loss_summary, acc_summary, lr_summary, sigma_summary])
train_summary_dir    = os.path.join(save_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

val_summary_op     = tf.summary.merge([loss_summary, acc_summary, sigma_summary])
val_summary_dir    = os.path.join(save_dir, "summaries", "val")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)


# restore pre-trained model
saver      = tf.train.Saver(tf.global_variables(), max_to_keep=100)
model_path = save_dir+'/models'
if iters>0:
    ckpt_path = model_path+'/ckpt-'+str(iters)
    
    saver.restore(sess, ckpt_path)
    print('Load model from {}'.format(ckpt_path))


# Train and Val stages
best_acc  = 0
best_loss = np.inf
wait      = 0

for ep in range(int(iters/100), n_epochs):
    loss_tr  = []
    acc_tr   = []
    loss_val = []
    acc_val  = []
    # run episodes training and then val
    for epi in tqdm(range(n_episodes), desc='train epoc:{}'.format(ep)):
        if ratio==1.0:
            support, s_labels, query, q_labels, _ = loader_train.next_data(n_way, n_shot, n_query)
        else:
            support, s_labels, query, q_labels, _ = loader_train.next_data_un(n_way, n_shot, n_query)

        _, summaries, step, ls, ac = sess.run([train_op, train_summary_op, global_step, ce_loss, acc], feed_dict={m.x: support, m.ys:s_labels, m.q: query, m.y:q_labels, m.phase:1})

        train_summary_writer.add_summary(summaries, step)
        loss_tr.append(ls)
        acc_tr.append(ac)

    # validation after each episode training, and decide if stop after train_patience steps
    for epi in tqdm(range(n_episodes), desc='val epoc:{}'.format(ep)):
        # validation to decide if stop
        support, s_labels, query, q_labels, _ = loader_val.next_data(n_test_way, n_test_shot, n_test_query, train=False)
        summaries, vls, vac = sess.run([val_summary_op, ce_loss, acc], feed_dict={m.x: support, m.ys:s_labels, m.q: query, m.y:q_labels, m.phase:0})

        val_summary_writer.add_summary(summaries, step)
        loss_val.append(vls)
        acc_val.append(vac)

    print('epoch:{}, loss:{:.5f}, acc:{:.5f}, val, loss:{:.5f}, acc:{:.5f}'.format(ep, np.mean(loss_tr), np.mean(acc_tr), np.mean(loss_val), np.mean(acc_val)))


    # Model save and stop criterion
    cond1 = (np.mean(acc_val)>best_acc)
    cond2 = (np.mean(loss_val)<best_loss)
    if cond1 or cond2:
        best_acc  = np.maximum(np.mean(acc_val),  best_acc)
        best_loss = np.minimum(np.mean(loss_val), best_loss)
        print('best val loss:{:.5f}, acc:{:.5f}'.format(best_loss, best_acc))
        
        # save the model
        saver.save(sess, model_path+'/ckpt', global_step=step)
        wait = 0
        
        f = open('checkpoints/'+args['exp_name']+'/log.txt', 'a')
        print('{} {:.5f} {:.5f}'.format(step, np.mean(loss_val), np.mean(acc_val)), file=f)
        f.close()
    else:
        wait += 1
        if ep%100==0:
            saver.save(sess, model_path+'/ckpt', global_step=step)
            f = open('checkpoints/'+args['exp_name']+'/log.txt', 'a')
            print('{} {:.5f} {:.5f}'.format(step, np.mean(loss_val), np.mean(acc_val)), file=f)
            f.close()

    if wait>patience and ep>n_epochs and rn>=0:
        break


