import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class ClassifierNet():
    def __init__(self, input_shape, num_classes, verbose = 1):
        self.model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
        ])

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        if (verbose):
            self.model.summary()

    
    def train(self, x_train, y_train, batch_size=128, epochs=1, validation_split=0.1, save = False, verbose = False):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    def evaluate(self, x_set, y_set, verbose=0):
        score = self.model.evaluate(x_set, y_set, verbose=0)
        return score



"""
fcs = []
for i in range(len(full_layers) - 1):
    fcs.append(nn.Linear(full_layers[i], full_layers[i + 1]).to(device))
"""