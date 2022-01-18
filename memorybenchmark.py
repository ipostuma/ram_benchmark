#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, BatchNormalization 
from tensorflow.keras.layers import Conv3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LeakyReLU, Reshape, Lambda
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import optimizers
#import keras
import numpy as np
import psutil
import humanize
import os
import argparse
import numpy as np

RESIDUAL = True
BASE_FILTER = 16
FILTER_GROW = True
DEEP_SUPERVISION = True
NUM_CLASS = 1

def diceloss(true,pred):
    acc=0
    elements_per_class=tf.math.reduce_sum(y_true)
    predicted_per_class=tf.math.reduce_sum(y_pred)
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred,y_true)))
    union=elements_per_class+predicted_per_class
    acc+=intersection/(union+0.000001)
    return 1.0-acc/2


def dice(true, pred):
    acc=0
    elements_per_class=tf.math.reduce_sum(y_true)
    predicted_per_class=tf.math.reduce_sum(y_pred)
    intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred,y_true)))
    union=elements_per_class+predicted_per_class
    acc+=intersection/(union+0.000001)
    return 1.0-acc/2

def myConv(x_in, nf, strides=1, kernel_size = 3):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    x_out = Conv3D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = BatchNormalization()(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

def Unet3dBlock(l, n_feat):
    if RESIDUAL:
        l_in = l
    for i in range(2):
        l = myConv(l, n_feat)
    return add([l_in, l]) if RESIDUAL else l


def UnetUpsample(l, num_filters):
    l = UpSampling3D()(l)
    l = myConv(l, num_filters)
    return l

def unet3d(vol_size, depth = 3):
    inputs = Input(shape=vol_size)
    filters = []
    down_list = []
    deep_supervision = None
    layer = myConv(inputs, BASE_FILTER)
    
    for d in range(depth):
        if FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        layer = Unet3dBlock(layer, n_feat = num_filters)
        down_list.append(layer)
        if d != depth - 1:
            layer = myConv(layer, num_filters*2, strides=2)
        
    for d in range(depth-2, -1, -1):
        layer = UnetUpsample(layer, filters[d])
        layer = concatenate([layer, down_list[d]])
        layer = myConv(layer, filters[d])
        layer = myConv(layer, filters[d], kernel_size = 1)
        
        if DEEP_SUPERVISION:
            if 0< d < 3:
                pred = myConv(layer, NUM_CLASS)
                if deep_supervision is None:
                    deep_supervision = pred
                else:
                    deep_supervision = add([pred, deep_supervision])
                deep_supervision = UpSampling3D()(deep_supervision)
    
    layer = myConv(layer, NUM_CLASS, kernel_size = 1)
    
    if DEEP_SUPERVISION:
        layer = add([layer, deep_supervision])
    layer = myConv(layer, NUM_CLASS, kernel_size = 1)
    x = Activation('softmax', name='softmax')(layer)
        
    model = Model(inputs=[inputs], outputs=[x])
    return model


def printm(filename):
    process = psutil.Process(os.getpid())
    f = open(filename,"a")
    f.write( humanize.naturalsize( psutil.virtual_memory().available ) + " " + humanize.naturalsize( process.memory_info().rss) + "\n")

def main():
    parser = argparse.ArgumentParser(description='some parameters to benchmark')
    parser.add_argument('shape', type=int,
                        help='shape of the image')
    parser.add_argument('depth', type=int,
                        help='depth of the 3D unet model')
    parser.add_argument('--epochs', type=int,
                        help='number of training epochs')
    args = parser.parse_args()

    epochs = 10
    if args.epochs:
        epochs = args.epochs

    s = args.shape
    xs = s
    ys = s
    zs = s
    depth = args.depth

    filename = "{}_{}_{}_{}".format(xs,ys,zs,depth)

    try:
        m = unet3d((xs,ys,zs,1),depth)
    except:
        return "model_error"

    printm(filename)

    # Open the file
    with open(filename + '_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        m.summary(print_fn=lambda x: fh.write(x + '\n'))


    x = np.random.rand(100,xs,ys,zs,1)
    y = np.ones((100,xs,ys,zs,1))

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.batch(1)

    printm(filename)

    adamlr = optimizers.Adam(
        learning_rate=0.00001, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-08, 
        amsgrad=True)

    log_dir = "logs/fit/{}".format(filename) 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '50,52')

    m.compile(
            loss=diceloss,
            optimizer=adamlr, 
            metrics=[dice])

    try:
        history=m.fit(
            dataset, 
            epochs=epochs,
            verbose=0,
            callbacks=[tensorboard_callback])
    except:
        return "fit_memory_error"
    printm(filename)

    return "ok"

if __name__ == "__main__":
    print(main())
