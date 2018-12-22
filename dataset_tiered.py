from   __future__ import print_function
import numpy as np
from   PIL import Image
import pickle as pkl
import os
import glob
import csv
from   tqdm import tqdm
import cv2

class dataset_tiered(object):
    def __init__(self, n_examples, n_episodes, split, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = n_examples
        self.n_episodes = n_episodes
        self.split      = split
        self.ratio      = args['ratio']
        self.seed       = args['seed']
        self.root_dir   = './data/tieredImagenet'

        self.iamge_data = []
        self.dict_index_label = []
        self.dict_index_unlabel = []
    


    def load_data_pkl(self):
        """
            load the pkl processed tieredImagenet into label,unlabel
            maintain label,unlabel data dictionary for indexes
        """
        labels_name = '{}/data/{}_labels.pkl'.format(self.root_dir, self.split)
        images_name = '{}/data/{}_images.npz'.format(self.root_dir, self.split)
        print('labels:',labels_name)
        print('images:',images_name)
        
        # decompress images if npz not exits
        if not os.path.exists(images_name):
            png_pkl = images_name[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(images_name, png_pkl)
            else:
                raise ValueError('path png_pkl not exits')
        
        if os.path.exists(images_name) and os.path.exists(labels_name):
            try:
                with open(labels_name) as f:
                    data            = pkl.load(f)
                    label_specific  = data["label_specific"]
                    #label_general  = data["label_general"]
                    #label_specific_str = data["label_specific_str"]
                    #label_general_str  = data["label_general_str"]
            except:
                with open(labels_name, 'rb') as f:
                    data            = pkl.load(f, encoding='bytes')
                    label_specific  = data[b'label_specific']
                    #label_general  = data[b"label_general"]
                    #label_specific_str = data[b"label_specific_str"]
                    #label_general_str  = data[b"label_general_str"]
            print('read label data:{}'.format(len(label_specific)))
        labels      = label_specific

        with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
            image_data = data["images"]    
            print('read image data:{}'.format(image_data.shape))
        
        n_classes   = np.max(labels)+1

        print('n_classes:{}, n_label:{}%, n_unlabel:{}%'.format(n_classes,self.ratio*100,(1-self.ratio)*100))
        dict_index_label    = {} # key:label, value:idxs
        dict_index_unlabel  = {}
        
        for cls in range(n_classes):
            idxs = np.where(labels==cls)[0]
            nums = idxs.shape[0]
            np.random.RandomState(self.seed).shuffle(idxs) # fix the seed to keep label,unlabel fixed

            n_label     = int(self.ratio*nums)
            n_unlabel   = nums-n_label

            dict_index_label[cls]   = idxs[0:n_label]
            dict_index_unlabel[cls] = idxs[n_label:]
    
        self.image_data         = image_data
        self.dict_index_label   = dict_index_label
        self.dict_index_unlabel = dict_index_unlabel
        self.n_classes          = n_classes
        print(dict_index_label[0])
        print(dict_index_unlabel[0])


    def next_data(self, n_way, n_shot, n_query, num_unlabel=0, n_distractor=0, train=True):
        """
            get support,query,unlabel data from n_way
            get unlabel data from n_distractor
        """
        support     = np.zeros([n_way, n_shot, self.im_height, self.im_width, self.channels], dtype=np.float32)
        query       = np.zeros([n_way, n_query, self.im_height, self.im_width, self.channels], dtype=np.float32)
        if num_unlabel>0:
            unlabel = np.zeros([n_way+n_distractor, num_unlabel, self.im_height, self.im_width, self.channels], dtype=np.float32)
        else:
            unlabel = []
            n_distractor = 0

        selected_classes = np.random.permutation(self.n_classes)[:n_way+n_distractor]
        for i, cls in enumerate(selected_classes[0:n_way]): # train way
            # labled data
            idx             = self.dict_index_label[cls]
            np.random.RandomState().shuffle(idx)
            idx1            = idx[0:n_shot + n_query]
            support[i]      = self.image_data[idx1[:n_shot]]
            query[i]        = self.image_data[idx1[n_shot:]]

            # unlabel
            if num_unlabel>0:
                idx         = self.dict_index_unlabel[cls]
                np.random.RandomState().shuffle(idx)
                idx2        = idx[0:num_unlabel]
                unlabel[i]  = self.image_data[idx2]
        
        for j,cls in enumerate(selected_classes[self.n_classes:]): # distractor way
            idx             = self.dict_index_unlabel[cls]
            np.random.RandomState().shuffle(idx)
            idx3            = idx[0:num_unlabel]
            unlabel[i+j]    = self.image_data[idx3]
        
        support_labels      = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels        = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        # unlabel_labels = np.tile(np.arange(n_way+n_distractor)[:, np.newaxis], (1, num_unlabel)).astype(np.uint8)
        return support, support_labels, query, query_labels, unlabel
    
    

def compress(path, output):
  with np.load(path, mmap_mode="r") as data:
    images = data["images"]
    array = []
    for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
      im = images[ii]
      im_str = cv2.imencode('.png', im)[1]
      array.append(im_str)
  with open(output, 'wb') as f:
    pkl.dump(array, f, protocol=pkl.HIGHEST_PROTOCOL)


def decompress(path, output):
  with open(output, 'rb') as f:
    array = pkl.load(f, encoding='bytes')
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)
