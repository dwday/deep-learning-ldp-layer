#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference:
Akgun, Devrim. "TensorFlow based deep learning layer for Local Derivative Patterns." 
Software Impacts 14 (2022): 100452.
https://www.sciencedirect.com/science/article/pii/S2665963822001361
"""
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from examples.test_models import test_model1,test_model2
import tensorflow as tf
#import numpy as np

# hyper parameters
batch_size  = 64
epochs      = 1000

lfw_people = fetch_lfw_people(min_faces_per_person=80, resize=1.0)
X=lfw_people.images

W=X.shape[1]
H=X.shape[2]
nchannel= 1 #gray image


# targets
y = lfw_people.target
#number of classses
nclasses = lfw_people.target_names.shape[0]


# Split dataset into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=42)


# display examples
plt.figure(1)
figs, axes = plt.subplots(4, 6)
for i in range(4):
    for j in range(6): 
        axes[i, j].imshow(x_train[i*6+j,:,:])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.show()


# normalization
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype( 'float32')/255.0

# convert integer labesls to categorical vectors
y_train = to_categorical(y_train,nclasses)
y_test  = to_categorical(y_test, nclasses)



# Test models: Model with ldp layer and baseline model with no ldp layer

# 1- Model with ldp layer ----------------------------------------------------
checkpoint_filepath1 = '/home/ubuntu/Documents/TEMP/checkpoints/checkpoint1.ckpt'
model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

#load model
model_ldp  = test_model1(W=W, H=H,nclass=nclasses,nchannel=nchannel,lr=1e-4)
model_ldp.summary()
    
# train model_ldp
history1 = model_ldp.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[model_checkpoint_callback1],
                    validation_split=0.2,
                    verbose=True)

# load model with best validation accuracy
model_ldp.load_weights(checkpoint_filepath1)

# evaluate model on the test dataset
test_loss1,test_acc1=model_ldp.evaluate(x_test,y_test)
print('test acc for model_ldp : ', test_acc1)

# 2- Baseline model -------------------------------------------------------
checkpoint_filepath2 = '/home/ubuntu/Documents/TEMP/checkpoints/checkpoint2.ckpt'
model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

#load model
model_base = test_model2(W=W, H=H,nclass=nclasses,nchannel=nchannel,lr=1e-4)
model_base.summary()

# train model_base
history2 = model_base.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[model_checkpoint_callback2],
                    validation_split=0.2,
                    verbose=True)

# load model with best validation accuracy
model_base.load_weights(checkpoint_filepath2)   
# evaluate model on the test dataset
test_loss2,test_acc2=model_base.evaluate(x_test,y_test)    
print('test acc for model_base: ', test_acc2)

# plot accuracy and loss graphs -----------------------------------------------
# model_ldp
acc1      = history1.history['acc']
val_acc1  = history1.history['val_acc']
loss1     = history1.history['loss']
val_loss1 = history1.history['val_loss']

# model_base
acc2      = history2.history['acc']
val_acc2  = history2.history['val_acc']
loss2     = history2.history['loss']
val_loss2 = history2.history['val_loss']        
    
plt.figure(2)
epochs = range(len(acc1))
plt.plot(epochs, acc1,     'g', label='Training acc - ldp')
plt.plot(epochs, val_acc1, 'b:',label='Validation acc - ldp')
plt.plot(epochs, acc2,     'r', label='Training acc - baseline')
plt.plot(epochs, val_acc2, 'm:',label='Validation acc - baseline')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.show()

plt.figure(3)
plt.plot(epochs, loss1,     'g',  label='Training loss - ldp')
plt.plot(epochs, val_loss1, 'b:', label='Validation loss - ldp')
plt.plot(epochs, loss2,     'r',  label='Training loss - baseline')
plt.plot(epochs, val_loss2, 'm:', label='Validation loss - baseline')
plt.title( 'Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()    

