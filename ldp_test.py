
# -*- coding: utf-8 -*-
"""
Example training  file using LDP layers

Akgun, Devrim. "TensorFlow based deep learning layer for Local Derivative Patterns." 
Software Impacts 14 (2022): 100452.
https://www.sciencedirect.com/science/article/pii/S2665963822001361
"""

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from examples.test_models import test_model1,test_model2
#from tensorflow.keras.utils import plot_model

nclass      = 10
batch_size  = 64
epochs      = 20
nchannel    = 3
W           = 32
H           = 32

# load train and test datasets 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# normalization
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype( 'float32')/255.0

# convert integer labesls to categorical vectors
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)


# Test models: Model with ldp layer and baseline model with no ldp layer 
model_ldp  = test_model1(W=W, H=H,nclass=nclass,nchannel=nchannel)
model_base = test_model2(W=W, H=H,nclass=nclass,nchannel=nchannel)

model_ldp.summary()
model_base.summary()


# train model_ldp
history1 = model_ldp.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))


test_loss1,test_acc1=model_ldp.evaluate(x_test,y_test)


# train model_base
history2 = model_base.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))


test_loss2,test_acc2=model_base.evaluate(x_test,y_test)

print('test acc for model_ldp : ', test_acc1)
print('test acc for model_base: ', test_acc2)


#plot training history 
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


# plot accuracy and loss graphs -----------------------------------------------
plt.figure()
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
plt.figure()
plt.plot(epochs, loss1,     'g',  label='Training loss - ldp')
plt.plot(epochs, val_loss1, 'b:', label='Validation loss - ldp')
plt.plot(epochs, loss2,     'r',  label='Training loss - baseline')
plt.plot(epochs, val_loss2, 'm:', label='Validation loss - baseline')
plt.title( 'Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.grid()