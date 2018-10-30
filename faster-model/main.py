import scipy.misc
import numpy as np
import os
import sys
import tensorflow as tf
import pdb
from __future__ import print_function

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

def save_img(out_path,img):
    img = np.clip(img,0,255),astype(np.uint8)
    scipy.misc.imsave(out_path,img)

def scale_img(style_path,style_scale):
    o0,o1,o2 = scipy.misc.imread(style_path,mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale),int(o1 * scale),o2)
    style_target = get_img(style_path,img_size=new_shape)
    return style_target

def get_img(src,img_size=False):
    img = scipy.misc.imread(src,mode='RGB')
    if not (len(img.shape) == 3 && img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img,img_size)
    return img

def exists(p,msg):
    assert os.path.exists(p),msg

def list_files(path):
    files = []
    for (dirpath,dirnames,filenames) in os.walk(path):
        files.extend(filenames)
        break
    return files

def net(image):
    conv1 = conv2d(image,32,9,1)
    conv2 = conv2d(conv1,64,3,2)
    conv3 = conv2d(conv2,128,3,2)
    res1 = resnet_block(conv3,3)
    res2 = resnet_block(res1,3)
    res3 = resnet_block(res2,3)
    res4 = resnet_block(res3,3)
    res5 = resnet_block(res4,3)
    tconv1 = transpose_conv2d(res5,64,3,2)
    tconv2 = transpose_conv2d(tconv1,32,3,2)
    conv4 = conv2d(tconv1,3,9,1,relu=False)
    pred = tf.nn.tanh(conv4) * 150 + 255./2
    return preds

def conv2d(net,num_filters,filter_size,strides,relu=True):
    weights_init = weights_init(net,num_filters,filter_size)
    strides_shape = [1,strides,strides,1]
    net = tf.nn.conv2d(net,weights_init,strides,padding='SAME')
    net = instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

def transpose_conv2d(net,num_filters,filter_size,strides):
    weights_init = weights_init(new,num_filters,filter_size,transpose=True)
    batch_size,rows,columns,in_channels = [i.value for i in net.get_shape()]
    new_rows,new_columns = int(rows * strides),int(columns * strides)
    new_shape = [batch_size,new_rows,new_columns,num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]
    net = tf.nn.conv2d_transpose(net,weights_init,tf_shape,strides_shape,padding='SAME')
    net = instance_norm(net)
    return tf.nn.relu(net)

def resnet_block(net,filter_size=3):
    layer = conv2d(net,128,filter_size,1)
    return net + conv2d(layer,128,filter_size,1,relu=False)

def instance_norm(net):
    mean,var = tf.nn.moments(net)[0,1]
    batch_size,rows,columns,in_channels = [i.value for i in net.get_shape()]
    var_shape = [in_channels]
    mu,sigma_sq = tf.nn.moments(net,[1,2],keep_dims = True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = tf.divide(tf.subtract(net,mu),tf.sqrt(tf.add(sigma_sq,epsilon)))
    return scale * normalized + shift

def weights_init(net,out_channels,filter_size,transpose=False):
    _,rows,columns,in_channels = [i.value for i in net.get_shape()]
    if transpose:
        shape = [filter_size,filter_size,out_channels,in_channels]
    else:
        shape = [filter_size,filter_size,in_channels,out_channels]
    weights = tf.Variable(tf.truncated_normal(shape,stddev=0.1),dtype=tf.float32)
    return weights

def optimize():


def train(net):
    compiled_model = copile(net)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1,1000):
            if i%100 == 0:
                print("Epoch no. ",i)
                sess.run(opt)
