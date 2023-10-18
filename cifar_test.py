#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example ldp application on Cifar10 using TensorFlow
Reference:
Akgun, Devrim. "TensorFlow based deep learning layer for Local Derivative Patterns." 
Software Impacts 14 (2022): 100452.
https://www.sciencedirect.com/science/article/pii/S2665963822001361
"""
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from pattern.layers import LDP


def tf_extract_ldp_rgb(x_train,mode='single', alpha='0'):
    [N,Rows,Cols,Channels]=x_train.shape  
    inx=layers.Input(shape=(Rows,Cols,3))      
    outx=LDP(mode=mode, alpha=alpha)(inx)     
    model=Model(inx,outx)
    model.compile()   
    x_train_ldp=model.predict(x_train,verbose=0)
    return x_train_ldp


# Load test dataset    
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#use first 100 images for test
x_train=x_train[:100,:,:,:]

   
# Extract lbp features for Cifar10 dataset using TensorFlow
start_time   = time.time() 

#process all images in the Cifar10 --------------------------------------------
x_train_ldp_0   = tf_extract_ldp_rgb( x_train, alpha='0'  ) 
x_train_ldp_45  = tf_extract_ldp_rgb( x_train, alpha='45' ) 
x_train_ldp_90  = tf_extract_ldp_rgb( x_train, alpha='90' ) 
x_train_ldp_135 = tf_extract_ldp_rgb( x_train, alpha='135') 
elapsed_tf = time.time() - start_time
print('tensor flow elapsed_time=',elapsed_tf)


# average of the features
x_train_ldp_mean = tf_extract_ldp_rgb( x_train, mode='mean') 
# alternatively
x_train_ldp_mean2=(x_train_ldp_0+x_train_ldp_45+x_train_ldp_90+x_train_ldp_135)/4.0
# error between the above results should be zero
err=np.sum(np.abs(x_train_ldp_mean2-x_train_ldp_mean))
print('mean features - error=',err)


# Example images---------------------------------------------------------------
#input images
plt.figure(1)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i, j].imshow(x_train[i*6+j,:,:,:])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])


# LDP images - average of 0, 45, 90 and 135
plt.figure(2)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i,j].imshow(x_train_ldp_mean[i*6+j,:,:,:])
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])


# LDP images - 0
plt.figure(3)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i,j].imshow(x_train_ldp_0[i*6+j,:,:,:])
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])        
        
        
# LDP images - 45
plt.figure(4)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i,j].imshow(x_train_ldp_45[i*6+j,:,:,:])
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])        
        
# LDP images - 90
plt.figure(4)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i,j].imshow(x_train_ldp_90[i*6+j,:,:,:])
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([]) 
        
# LDP images - 135
plt.figure(4)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i,j].imshow(x_train_ldp_135[i*6+j,:,:,:])
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([]) 
