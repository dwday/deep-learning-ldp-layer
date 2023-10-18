# -*- coding: utf-8 -*-
"""
Example test models 

"""
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model
from pattern.layers import LDP


# 1- Model with ldp layer ----------------------------------------------------
def test_model1(W=32, H=32, nclass=10, nchannel=3,lr=1e-4):
    in1 = layers.Input(shape=(W, H,nchannel))
    x = layers.Conv2D(8, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(in1)
    #--------------------------------------
    x1 = LDP(mode='single', alpha='0' )(x)
    x2 = LDP(mode='single', alpha='45')(x)
    x3 = LDP(mode='single', alpha='90')(x)
    x4 = LDP(mode='single', alpha='135')(x)

    x1 = layers.Add()([x1,x])
    x2 = layers.Add()([x2,x])
    x3 = layers.Add()([x3,x])
    x4 = layers.Add()([x4,x])
    
    x = layers.Concatenate()([x1,x2,x3,x4])
    #--------------------------------------
    x = layers.MaxPool2D((2, 2))(x)
    x=layers.BatchNormalization()(x)    
   
    x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)
    x=layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    

    x = layers.Dropout(0.6)(x)

    # x = layers.Dense(128, activation='relu')(x)

    output = layers.Dense(nclass, activation='softmax')(x)

    model = Model(inputs=in1, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=lr),
                  metrics=['acc'])
    return model

# 2- Baseline model -------------------------------------------------------
def test_model2(W=32, H=32, nclass=10, nchannel=3,lr=1e-4):

    in1 = layers.Input(shape=(W, H, nchannel))

    x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                      padding='same',
                      activation='relu')(in1)
    
    x = layers.MaxPool2D((2, 2))(x)
    x=layers.BatchNormalization()(x)    
    x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)
    
    x=layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      padding='valid',
                      activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    

    x = layers.Dropout(0.6)(x)

    # x = layers.Dense(128, activation='relu')(x)

    output = layers.Dense(nclass, activation='softmax')(x)

    model = Model(inputs=in1, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=lr),
                  metrics=['acc'])
    return model