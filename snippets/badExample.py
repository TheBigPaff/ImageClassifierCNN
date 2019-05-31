# libraries ...
# getting data sets ...


# BUILDS AND TRAIN MY OWN CNN
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
