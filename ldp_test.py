
# -*- coding: utf-8 -*-
"""
Example training  file using LDP layers
*Cifar10 dataset has been used  the for toy example 
*Replace with your model and dataset
"""


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from examples.test_models import test_model1,test_model2
#from tensorflow.keras.utils import plot_model



W = 32
H = 32
nclass = 10
batch_size = 64
epochs = 20

# load test model
model = test_model1(W=W, H=H,nclass=nclass)
# model = test_model2(W=W, H=H,nclass=nclass)
model.summary()


# load train and test datasets 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# normalization
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# convert integer labesls to categorical vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# train model
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))

#plot training history
# getaccurracy and loss values
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plot accuracy and loss graphs
plt.figure()
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()