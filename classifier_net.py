import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random

class ClassifierNet():
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, layer_dimensions=[70], verbose = 0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layer_dimensions = layer_dimensions

        self.instantiate_model()

        if (verbose):
            self.model.summary()
    
    def instantiate_model(self):
        input_layer = [keras.Input(shape=self.input_shape), layers.Flatten()]
        inner_layers = [layers.Dense(num_units, activation="relu") for num_units in self.layer_dimensions]
        output_layer = [layers.Dense(self.num_classes, activation="softmax")]

        sequential_layers = input_layer + inner_layers + output_layer

        self.model = keras.Sequential(sequential_layers)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    def mutate_layout(self, new_layer_dimensions):
        num_layers = len(self.layer_dimensions)
        layer_index = random.randint(0, num_layers)
        layer_dimensions = [random.randint(1, 11)*10 for i in range(num_layers)]
        
        self.layer_dimensions = new_layer_dimensions
        self.instantiate_model()
    
    def train(self, x_train, y_train, batch_size=128, epochs=1, validation_split=0.1, save = False, verbose = False):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    def evaluate(self, x_set, y_set, verbose=0):
        score = self.model.evaluate(x_set, y_set, verbose=0)
        return score