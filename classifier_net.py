import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random

from tensorflow.keras.preprocessing.sequence import pad_sequences

class ClassifierNet():
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, layer_dimensions=[3, 3, 3], verbose = 0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layer_dimensions = layer_dimensions

        self.max_layer_size = 5
        self.input_layer_size = 784

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
    
    def mutate_layout(self):
        num_layers = len(self.layer_dimensions)
        new_layer_index = random.randint(0, num_layers)
        new_layer_size = random.randint(int(self.max_layer_size / 3), self.max_layer_size)

        self.layer_dimensions[new_layer_index] = new_layer_size
        self.instantiate_model()
    
    def train(self, x_train, y_train, batch_size=128, epochs=1, validation_split=0.1, save = False, verbose = False):
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    def evaluate(self, x_set, y_set, verbose=0):
        score = self.model.evaluate(x_set, y_set, verbose=0)
        return score
    
    def get_training_data(self):
        return self.history.history #self.model.history.history
    
    def get_weights(self):
        model_weights = np.array([])
        for lay in self.model.layers:
            print(lay.name)
            layer_weights = lay.get_weights()
            if len(layer_weights) > 0:
                if layer_weights[0].shape[0] > self.max_layer_size:
                    # If input layer
                    layer_width = self.input_layer_size
                    total_width = self.input_layer_size*self.max_layer_size + self.max_layer_size
                else:
                    # If not input layer
                    layer_width = self.max_layer_size
                    total_width = self.max_layer_size*self.max_layer_size + self.max_layer_size

                weights = layer_weights[0]
                padded_weights = pad_sequences(weights, maxlen=self.max_layer_size, padding='post', dtype='float32')

                extra_padding = np.array([np.zeros(self.max_layer_size) for i in range(layer_width - len(weights))])
                if (extra_padding.shape[0] > 0):
                    padded_weights = np.concatenate((padded_weights, extra_padding))

                padded_weights = [padded_weights.flatten()]
                padded_weights = pad_sequences(padded_weights, maxlen=total_width, padding='post', dtype='float32')

                biases = [layer_weights[1]]
                padded_biases = pad_sequences(biases, maxlen=self.max_layer_size, padding='post', dtype='float32')
                padded_biases = pad_sequences(padded_biases, maxlen=total_width, padding='pre', dtype='float32')

                combined = padded_weights[0] + padded_biases[0]
                model_weights = np.concatenate((model_weights, combined))
        return model_weights
    
    def load_from_prediction(self, prediction):
        input_width = self.input_layer_size*self.max_layer_size + self.max_layer_size
        middle_width = self.max_layer_size*self.max_layer_size + self.max_layer_size

        print("K")