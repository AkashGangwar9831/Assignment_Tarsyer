# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:18:56 2020

@author: Akash
"""

from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Scale dataset values to lie between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

'''
Add Noise to our MNNIST Dataset by sampling random values from Gaussian distribution by using 
np.random.normal() and adding it to our original images to change pixel values
'''
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


#Visualising the noisy digits using Matplotlib
n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#Specify the input layer size which is 28x28x1
input_img = Input(shape=(28, 28, 1))


#Model construction
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


#At this point the representation is (7, 7, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train_noisy, x_train,epochs=100,batch_size=128,shuffle=True,validation_data=(x_test_noisy, x_test)).history


#Plotting the graph for training and validation loss 
plt.plot(history['loss'], color='darkblue', alpha=0.85)
plt.plot(history['val_loss'], color='darkgreen', alpha=0.85)
plt.title('Model Loss', fontsize=20, weight='bold')
plt.ylabel('loss', fontsize=10, weight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('epoch', fontsize=10, weight='bold')
plt.legend(['train', 'test'], loc='upper right', fontsize=10)