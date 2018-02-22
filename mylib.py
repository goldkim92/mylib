#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:46:21 2017

@author: jm
"""

#%% import
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.misc
from PIL import Image

#%% test function
def fox_say():
    print("Woooooh")

#%% cv2 image show
def imshowOnConsole(img,figsize=None):
    plt.figure(figsize=figsize)
#    if img is not in the form of unit8, imshow gives weird results
    fig = plt.imshow(np.uint8(img))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    
def cv2imshow(img, figsize=None):
    if len(img.shape)==3 and img.shape[2]==3:
        b, g, r = np.split(img, 3, axis=2)
        img = np.concatenate([r,g,b], axis=2)
        imshowOnConsole(img,figsize)
    else:
        imshowOnConsole(img,figsize)
        
#%% using scipy.misc
def imread(path):
    img = scipy.misc.imread(path) #.astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)    
    
def imresize(img, size):
#    size : int -> percentage of current size
#    size : float -> fraction of current size
#    size : tuple -> size of the output image
    return scipy.misc.imresize(img, size)

# it throws RuntimeError after closing the windows....
# However, this is useful to see the coordination and real-value of RGB
def imshowOnImage(arr):
    scipy.misc.imshow(arr)
    
def show_result(batch_res, fname, img_height=28, img_width=28, grid_size=(8, 8), grid_pad=5):
    #img_height = img_width = 28 for MNIST
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)
  
# load train data with augmentation by resize and crop, flip
def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0]) #horse
    img_B = imread(image_path[1]) #zebra
    if not is_testing: # training
        # this is for Augmentation : JM
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size))) # 1<= h1 <= 30
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size))) # 1<= h1 <= 30
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else: # testing
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1. # -1 <= img_A <= 1
    img_B = img_B/127.5 - 1. # -1 <= img_A <= 1

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

#%% MNIST accuracy
def t_accuracy(sess,t='train'):
    batch_size = 100
    test_accuracy = 0.
    
    if t=='test':
        dataset = mnist.test
    else:
        dataset = mnist.train
    batch_idxs = dataset.num_examples/batch_size
    
    for _ in range(int(batch_idxs)):
        batch = dataset.next_batch(batch_size)
        test_accuracy += sess.run(accuracy,feed_dict={x:batch[0],true_y:batch[1]}) / batch_idxs 
    return test_accuracy


#%% file and directory (csv file)
import os
from tqdm import tqdm
from csv import DictWriter

# make directory if not exist
try: os.makedirs('nmt_news')
except: pass

# make file if not exist
try: open('nmt_news/stances.csv','x')
except FileExistsError: pass

open_mode = 'w' # w: write, a: append
with open('news_dataset.csv',open_mode, encoding='UTF-8',newline='') as f:
    w = DictWriter(f, ['식별자', '뉴스제목', '뉴스본문','Label','Link'])
    if open_mode=='w':
        w.writeheader()
    for i in tqdm(range(len(articles))):
        k_dict = {
                '식별자': (i+1),
                '뉴스제목': titles[i],
                '뉴스본문':articles[i],
                'Label':0,
                'Link':links[i]
                }
        w.writerow(k_dict)

#%% file and directory (csv txt file)
path = os.path.join('CelebA','bbox.txt')
file = open(path,'r')

bbox = []

n = file.readline()
n = int(n.split('\n')[0]) # for celebA/bbox : 202599

attr_line = file.readline()
attrs = attr_line.split('\n')[0].split()

ATTRS = namedtuple('ATTRS', [attr for attr in attrs])


for line in file:
    row = line.split('\n')[0].split()
    row = ATTRS(row[0],row[1],row[2],row[3],row[4])
    bbox.append(row)

file.close()


#%% show all images in html file

# make file if not exist
index_path = 'index.html'
try: open(index_path,'x')
except FileExistsError: pass

#open file
index = open(index_path, "w")
index.write("<html><body><table><tr>")
index.write("<th>name</th><th>output</th></tr>")

# get fake image
data_paths = glob.glob(path)
data_paths.sort(key=natural_keys)

print('Processing image')
for data_path in data_paths:
    index.write("<td>%s</td>" % os.path.split(data_path)[1])
    index.write("<td><img src='%s'></td>" % data_path)
    index.write("</tr>")
index.close()

#%%
# Sort a string with a number inside
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

alist=[
    "something1",
    "something2",
    "something1.0",
    "something1.25",
    "something1.105"]

alist.sort(key=natural_keys)
print(alist)
