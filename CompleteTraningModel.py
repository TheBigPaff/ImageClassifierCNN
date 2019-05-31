import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

TRAIN_DIR = r'cats-dogs/train'
TEST_DIR = r'cats-dogs/test'

image_target_size = (224, 224)
# needs to format data for the model
train_batches = ImageDataGenerator().flow_from_directory(TRAIN_DIR, target_size=image_target_size,
                                                         classes=['dog', 'cat'], batch_size=25)
test_batches = ImageDataGenerator().flow_from_directory(TEST_DIR, target_size=image_target_size,
                                                        classes=None, batch_size=10)


"""
# BUILD FINE-TUNED VGG16 MODEL
vgg16_model = keras.applications.vgg16.VGG16()  # getting VGG16 model from the keras library for pre-trained models
# vgg16_model.summary()

# we need to transform the model into a sequential object (so we can work with it and fine-tune it)
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False  # we need to freeze every layers so the weight will not be updated.
    # This is useful for fine-tuning
model.add(Dense(2, activation='softmax'))  # now let's add back an output layer but now with an output shape of 2.

# compiling model (setting optimizer and loss function)
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
"""

from keras.models import load_model
model = load_model('fine_tuned_vgg16_model.h5')
# TRAIN THE FINE-TUNED VGG16 MODEL
model.fit_generator(train_batches, steps_per_epoch=8, epochs=10, verbose=2)
model.save('fine_tuned_vgg16_model.h5')
