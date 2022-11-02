#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example test models using LDP transform

"""
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model
from pattern.layers import LDP

def test_model1(W=32, H=32,nclass=10):
    
    in1 = layers.Input(shape=(W, H, 3))

    x1 = layers.Conv2D(8, (1, 1), strides=(1, 1),
                       padding='valid',
                       activation='relu')(in1)
    
    x2 = layers.Conv2D(8, (1, 1), strides=(1, 1),
                       padding='valid',
                       activation='relu')(in1)
    
    x3 = layers.Conv2D(8, (1, 1), strides=(1, 1),
                       padding='valid',
                       activation='relu')(in1)
    
    x4 = layers.Conv2D(8, (1, 1), strides=(1, 1),
                       padding='valid',
                       activation='relu')(in1)
    
    x1 = LDP(mode='single', alpha='0')(x1)
    
    x2 = LDP(mode='single',alpha='45')(x2)
    
    x3 = LDP(mode='single',alpha='90')(x3)
    
    x4 = LDP(mode='single',alpha='135')(x4)
  
    x = layers.Concatenate()([x1,x2,x3,x4])
    
  
    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)
    

    x = layers.MaxPool2D((2,2))(x)
    
    
    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)

    x = layers.MaxPool2D((2,2))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128,activation='relu')(x)
 
    output = layers.Dense(nclass, activation='softmax')(x)
    
    model = Model(inputs=in1, outputs=output)
    
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['acc'])
    return model


def test_model2(W=32, H=32,nclass=10):
    
    in1 = layers.Input(shape=(W, H, 3))

    x = layers.Conv2D(8, (1, 1), strides=(1, 1),
                       padding='valid',
                       activation='relu')(in1)

    
    x = LDP(mode='multi')(x)
    
  
    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)
    

    x = layers.MaxPool2D((2,2))(x)
    
    
    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)

    x = layers.MaxPool2D((2,2))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128,activation='relu')(x)
 
    output = layers.Dense(nclass, activation='softmax')(x)
    
    model = Model(inputs=in1, outputs=output)
    
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['acc'])
    return model