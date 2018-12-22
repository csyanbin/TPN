# coding: utf-8
from   __future__ import print_function
import numpy as np
import tensorflow as tf
from   tensorflow.contrib.layers.python import layers as tf_layers

class models(object):
    def __init__(self, args):
        # parameters
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
        self.args              = args

        # placeholders for data and label
        self.x      = tf.placeholder(tf.float32, [None, None, self.im_height, self.im_width, self.channels])
        self.ys     = tf.placeholder(tf.int64, [None, None])
        self.q      = tf.placeholder(tf.float32, [None, None, self.im_height, self.im_width, self.channels])
        self.y      = tf.placeholder(tf.int64, [None, None])
        self.phase  = tf.placeholder(tf.bool, name='phase')
        
        self.alpha  = args['alpha']


    def conv_block(self, inputs, out_channels, pool_pad='VALID', name='conv'):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding="same")
            conv = tf.contrib.layers.batch_norm(conv, is_training=self.phase, decay=0.999, epsilon=1e-3, scale=True, center=True)
            conv = tf.nn.relu(conv)
            conv = tf.contrib.layers.max_pool2d(conv, 2, padding=pool_pad)

            return conv


    def encoder(self, x, h_dim, z_dim, reuse=False):
        """Feature embedding network"""
        with tf.variable_scope('encoder', reuse=reuse):
            net = self.conv_block(x,   h_dim, name='conv_1')
            net = self.conv_block(net, h_dim, name='conv_2')
            net = self.conv_block(net, h_dim, name='conv_3')
            net = self.conv_block(net, z_dim, name='conv_4')
            
            net = tf.contrib.layers.flatten(net)

            return net


    def relation(self, x, h_dim, z_dim, reuse=False):
        """Graph Construction Module"""
        with tf.variable_scope('relation', reuse=reuse):
            x   = tf.reshape(x, (-1,5,5,64))

            net = self.conv_block(x,    h_dim,   pool_pad='SAME', name='conv_5')
            net = self.conv_block(net,  1,       pool_pad='SAME', name='conv_6')
        
            net = tf.contrib.layers.flatten(net)
            
            net = tf.contrib.layers.fully_connected(net, 8)
            net = tf.contrib.layers.fully_connected(net, 1, tf.identity)
            
            net = tf.contrib.layers.flatten(net)

            return net


    # contruct the model
    def construct(self):
        # data input
        x_shape                  = tf.shape(self.x)
        q_shape                  = tf.shape(self.q)
        num_classes, num_support = x_shape[0], x_shape[1]
        num_queries              = q_shape[1]

        ys_one_hot = tf.one_hot(self.ys, depth=num_classes)
        y_one_hot  = tf.one_hot(self.y,  depth=num_classes)

        # construct the model
        x       = tf.reshape(self.x, [num_classes * num_support, self.im_height, self.im_width, self.channels])
        q       = tf.reshape(self.q, [num_classes * num_queries, self.im_height, self.im_width, self.channels])
        emb_x   = self.encoder(x, self.h_dim, self.z_dim)
        emb_q   = self.encoder(q, self.h_dim, self.z_dim, reuse=True)
        emb_dim = tf.shape(emb_x)[-1]

        if self.args['rn']==300:       # learned sigma, fixed alpha
            self.alpha = tf.constant(self.args['alpha'])
        else:                          # learned sigma and alpha
            self.alpha = tf.Variable(self.alpha, name='alpha')

        ce_loss, acc, sigma_value = self.label_prop(emb_x, emb_q, ys_one_hot)

        return ce_loss, acc, sigma_value

    
       
    def label_prop(self, x, u, ys):

        epsilon = np.finfo(float).eps
        # x: NxD, u: UxD
        s       = tf.shape(ys)
        ys      = tf.reshape(ys, (s[0]*s[1],-1))
        Ns, C   = tf.shape(ys)[0], tf.shape(ys)[1]
        Nu      = tf.shape(u)[0]
        
        yu      = tf.zeros((Nu,C))/tf.cast(C,tf.float32) + epsilon  # 0 initialization
        #yu = tf.ones((Nu,C))/tf.cast(C,tf.float32)            # 1/C initialization
        y       = tf.concat([ys,yu],axis=0)
        gt      = tf.reshape(tf.tile(tf.expand_dims(tf.range(C),1), [1,tf.cast(Nu/C,tf.int32)]), [-1])

        all_un  = tf.concat([x,u],0)
        all_un  = tf.reshape(all_un, [-1, 1600])
        N, d    = tf.shape(all_un)[0], tf.shape(all_un)[1]
        
        # compute graph weights
        if self.args['rn'] in [30, 300]:   # compute example-wise sigma
            self.sigma = self.relation(all_un, self.h_dim, self.z_dim)

            all_un  = all_un / (self.sigma+epsilon)
            all1    = tf.expand_dims(all_un, axis=0)
            all2    = tf.expand_dims(all_un, axis=1)
            W       = tf.reduce_mean(tf.square(all1-all2), axis=2)
            W       = tf.exp(-W/2)

        # kNN Graph
        if self.args['k']>0:
            W = self.topk(W, self.args['k'])
        
        # Laplacian norm
        D           = tf.reduce_sum(W, axis=0)
        D_inv       = 1.0/(D+epsilon)
        D_sqrt_inv  = tf.sqrt(D_inv)

        # compute propagated label
        D1          = tf.expand_dims(D_sqrt_inv, axis=1)
        D2          = tf.expand_dims(D_sqrt_inv, axis=0)
        S           = D1*W*D2
        F           = tf.matrix_inverse(tf.eye(N)-self.alpha*S+epsilon)
        F           = tf.matmul(F,y)
        label       = tf.argmax(F, 1)

        # loss computation
        F = tf.nn.softmax(F)
        
        y_one_hot   = tf.reshape(tf.one_hot(gt,depth=C),[Nu, -1])
        y_one_hot   = tf.concat([ys,y_one_hot], axis=0)
        
        ce_loss     = y_one_hot*tf.log(F+epsilon)
        ce_loss     = tf.negative(ce_loss)
        ce_loss     = tf.reduce_mean(tf.reduce_sum(ce_loss,1))
        
        # only consider query examples acc
        F_un        = F[Ns:,:]
        acc         = tf.reduce_mean(tf.to_float(tf.equal(label[Ns:],tf.cast(gt,tf.int64))))

        return ce_loss, acc, self.sigma

    
    def topk(self, W, k):
         # construct k-NN and compute margin loss
        values, indices     = tf.nn.top_k(W, k, sorted=False)
        my_range            = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1) 
        my_range_repeated   = tf.tile(my_range, [1, k])
        full_indices        = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], axis=2)
        full_indices        = tf.reshape(full_indices, [-1, 2])
        
        topk_W  = tf.sparse_to_dense(full_indices, tf.shape(W), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
        ind1    = (topk_W>0)|(tf.transpose(topk_W)>0) # union, k-nearest neighbor
        ind2    = (topk_W>0)&(tf.transpose(topk_W)>0) # intersection, mutal k-nearest neighbor
        ind1    = tf.cast(ind1,tf.float32)

        topk_W  = ind1*W
        
        return topk_W
    


