from pyeasyga import pyeasyga
import numpy as np
import random

from classifier_net import ClassifierNet
from predictor import PredictorNet

class GeneticFit():
    def __init__(self, training_set, validation_set, num_epochs=1, max_iter=3, pop_size=2, num_generations=2):
        self.training_set = training_set
        self.validation_set = validation_set
        
        self.num_epochs = num_epochs
        self.max_iter = max_iter

        self.pop_size = pop_size
        self.num_generations = num_generations

        self.seen = 0
        self.gens = 0
    
    def fitness(self, individual, data):
        print("####################")
        print("Net has dimensions: ", individual.layers)
        fitness = individual.evaluate(self.validation_set[0], self.validation_set[1])

        self.seen += 1
        
        if (self.seen == self.pop_size):
            self.gens += 1
            print("###############################################################")
            print("Generation ", self.gens, " complete")
            print("###############################################################")
            self.seen = 0

        return fitness
    
    def create_individual(self, data):
        # Generate random number of layers and dimensions
        num_layers = random.randint(1, 2)
        layer_dimensions = [random.randint(1, 11)*10 for i in range(num_layers)]
        print("layer dimensions are", layer_dimensions)

        individual = ClassifierNet(layer_dimensions=layer_dimensions)

        individual.train(self.training_set[0], self.training_set[1])
        return individual
    
    def mutate(self, individual):
        # randomly mutate the number of layers or the size of a layer
        individual.mutate_layout()
    
    def crossover(self, parent_1, parent_2):
        # get index of shorter parent to ensure crossover can occur
        """
        min_parent_size = np.min([len(parent_1), len(parent_2)])
        crossover_index = random.randint(1, min_parent_size)

        child_1_layers = parent_1[:crossover_index] + parent_2[crossover_index:]
        child_2_layers = parent_2[:crossover_index] + parent_1[crossover_index:]
        """
        return parent_1, parent_2 #child_1, child_2
    
    def geneticFit(self, pretrained = False):
        data = self.training_set

        ga = pyeasyga.GeneticAlgorithm(data,
                               population_size=self.pop_size,
                               generations=self.num_generations,
                               crossover_probability=0.0,
                               mutation_probability=0.5,
                               elitism=True,
                               maximise_fitness=True)

        ga.create_individual = self.create_individual
        ga.fitness_function = self.fitness
        ga.mutate_function = self.mutate
        ga.crossover_function = self.crossover

        ga.run()

        print("Best individual is: ")
        print(ga.best_individual()[1].layer_dimensions)
        print("With performance: ", ga.best_individual()[1].evaluate(self.validation_set[0], self.validation_set[1]))
        print("Saving model...")

        BEST_PATH = './best_model.pth'
        torch.save(ga.best_individual()[1].state_dict(), BEST_PATH)

        return ga.best_individual()