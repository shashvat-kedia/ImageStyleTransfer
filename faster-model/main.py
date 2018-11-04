import scipy.misc
import numpy as np
import os
import sys
import tensorflow as tf
import pdb
from __future__ import print_function
from operator import mul
import random
import time

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

def instance_norm(net,epsilon=1e-3):
    batch_size,rows,columns,in_channels = [i.value for i in net.get_shape()]
    var_shape = [in_channels]
    mu,sigma_sq = tf.nn.moments(net,[1,2],keep_dims = True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
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

def optimize(content_targets,style_target,content_weight,style_weight,tv_weight,vgg_path,batch_size,epochs=2,slow=False):
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        content_targets = content_targets[:-mod]
    style_features = {}
    shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)
    with tf.Graph().as_default(),tf.Session() as sess:
        style_image = tf.placeholder(tf.float32,shape=style_shape)
        style_image_processed = vgg.preprocess(style_image)
        net = vgg.net(vgg_path,style_image_processed)
        style_pre_pro = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:})
            features = np.reshape(features,(-1,features.shape[3]))
            gram = np.matmul(features.T,features) / features.size
            style_features[layer] = gram
    with tf.Graph().as_default(),tf.Session() as sess:
        content_image = tf.placeholder(tf.float32,shape=shape)
        contet_image_preprocesses = vgg.preprocess(content_image)
        content_features = {}
        content_net = vgg.net(vgg_path,content_image_processed)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        if slow:
            preds = tf.Variable(tf.random_normal(content_image.get_shape())) * 0.256
            preds_pre = preds
        else:
            preds = net(content_image/255.0)
            preds_pre = vgg.preprocess(preds)
        net = vgg.net(vgg_path,preds_pre)
        content_size = tensor_size(content_eatures[CONTENT_LAYER]) * batch_size
        assert tensor_size(content_features[CONTENT_LAYER]) == tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER])/content_size)
        style_losses = []
        for layer in STYLE_LAYERS:
            style_layer = net[layer]
            batch_size,rows,columns,in_channels = [i.value for i in style_layer.get_shape()]
            size = rows * columns * in_channels
            feats = tf.reshape(style_layer,(batch_size,rows*columns,in_channels))
            feats_T = tf.transpose(feats,perm=[0,2,1])
            grams = tf.matmul(feats_T,feats)/size
            style_gram = style_features[layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
        style_loss = style_weight * functools.reduce(tf.add,style_losses)/batch_size
        tv_y_size = tensor_size(preds[:,1:,:,:])
        tv_x_size = tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:shape[2]-1,:])
        tv_loss = tv_weight * (x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        loss = content_loss + style_loss + tv_loss
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        sess.run(tf.global_variables_initialier())
        uid = random.randint(1,100)
        print("UID: ",uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterationis * batch_size
                step = curr + batch_size
                x_batch = np.zeros(shape,dtype=np.float32)
                for j,img_p in enumerate(content_targets[curr:step]):
                    x_batch[j] = get_img(img_p,(256,256,3)).astype(np.float32)
                iteration += 1
                assert x_batch.shape[0] == batch_size
                feed_dict = {
                x_content:x_batch
                }
                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                is_print_iter = int(iterations) % 1000 == 0
                if slow:
                    is_print_iter = int(epochs) % 1000 == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss,content_loss,tv_loss,loss,preds]
                    test_feed_dict = {
                    x_content:x_batch
                    }
                    tup = sess.run(to_get,feed_dict=test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss,_content_loss,_tv_loss,_loss)
                    if slow:
                        _preds = vgg.unprocess(_preds)
                    else:
                        saver = tf.train.Saver()
                        res = saver.save(sess,save_path)
                    yield(_preds,losses,iterations,epoch)


def tensor_size(tensor):
    return functools.reduce(mul,(d.value for d in tensor.get_shape()[1:]),1)

def main():
    slow = 1
    kwargs = {
    "slow": slow,
    "epochs": 100,
    "print_iterations": 10,
    "batch_size": 10,
    "learning_rate": 1e-3
    }
    if slow:
        if kwargs["epochs"] < 10:
            kwargs["epochs"] = 1000
        if kwargs["learning_rate"] < 1:
            kwargs["learning_rate"] = 1e1
    args = {
    content_targets,
    style_targets,
    content_weight,
    style_weight,
    tv_weight,
    vgg_path
    }
