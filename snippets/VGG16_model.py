"""
# ----------------------------------- BUILDS/TRAINS/PREDICTS MY OWN CNN ------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    Flatten(),
    Dense(2, activation='softmax')
])
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# train data generated batch by batch
model.fit_generator(train_batches, steps_per_epoch=5, epochs=10, verbose=2)

# predicting
predictions = model.predict_generator(test_batches, steps=1, verbose=2)
print(predictions)
# ----------------------------------------------------------------------------------------------------------------------
"""


# BUILD FINE-TUNED VGG16 MODEL
vgg16_model = keras.applications.vgg16.VGG16()  # getting VGG16 model from the keras library for pre-trained models
# vgg16_model.summary()

# we need to transform the model into a sequential object (so we can work with it and fine-tune it)
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
# model.summary() ---> it will be the same as vgg16_model

model.layers.pop()  # now let's remove the dense output layers of 1000 nodes (we just need 2)
for layer in model.layers:
    layer.trainable = False  # we need to freeze every layers so the weight will not be updated.
    # This is useful for fine-tuning
model.add(Dense(2, activation='softmax'))  # now let's add back an output layer but now with an output shape of 2.

# compiling model (setting optimizer and loss function)
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN THE FINE-TUNED VGG16 MODEL
model.fit_generator(train_batches, steps_per_epoch=8, epochs=10, verbose=2)
