import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import rendering

from classifier_net import ClassifierNet

# DEFINE ENVIRONMENT
class PredictorEnvironment(gym.Env):
    """
    Custom environment implementing openAI gym interface
    """
    def __init__(self, training_set, validation_set):
        # Define action and observation space
        # Set parameters
        MAX_LAYERS = 3
        MAX_UNITS = 100

        # Set up observation
        N_CONTINUOUS_OBSERVATIONS = 420
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, N_CONTINUOUS_OBSERVATIONS))

        # Set up action space
        N_CONTINUOUS_ACTIONS = 3*420
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_CONTINUOUS_ACTIONS,))


        # Set up data
        self.training_set = training_set
        self.validation_set = validation_set

        self.maxlen = 100
        self.trained = False


        self.individual= ClassifierNet()
        print("Training base:")
        self.individual.train(self.training_set[0], self.training_set[1], epochs=1)

        self.default_individual = ClassifierNet(layer_dimensions=self.individual.layer_dimensions)

        # Mutate classifier net
        self.default_individual.mutate_layout()

        print("Training default initialization:")
        self.default_individual.train(self.training_set[0], self.training_set[1], epochs=1)
    
    def reset(self):
        return self.get_state()
    
    def compute_reward(self, default_training_data, pred_training_data):
        default_losses = np.array(default_training_data['loss'])
        pred_losses = np.array(pred_training_data['loss'])

        #performance_gain = np.sum(default_losses - pred_losses)
        return -10*(np.sum(np.array(pred_training_data['loss'])))**3#performance_gain #pred_losses if performance_gain < 0 else performance_gain
    
    def step(self, pred_weights):

        pred_individual = ClassifierNet(layer_dimensions=self.default_individual.layer_dimensions)
        pred_individual.load_from_lstm_prediction(pred_weights)
    
        #print("Training default initialization:")
        #self.default_individual.train(self.training_set[0], self.training_set[1])

        print("Training RL initialization:")
        pred_individual.train(self.training_set[0], self.training_set[1])

        reward = self.compute_reward(self.default_individual.get_training_data(), pred_individual.get_training_data())
        print("Reward: ", reward)
        observations = self.get_state()
        done = True
        info = {}
        return observations, reward, done, info
    
    def get_state(self):
        old_individual_weights = self.individual.get_lstm_weights()#.flatten()
        new_individual_weights = self.default_individual.get_lstm_weights()
        new_individual_weights[new_individual_weights > 0.0] = 1.0

        curr_state = old_individual_weights #np.concatenate((old_individual_weights, new_individual_weights))
        return curr_state

    def render(self, mode='human'):
        print("Render not implemented")
    
    def close(self):
        print("Close not implemented")
    
    def mutate_for_supervised(self):
        num_layers = len(self.layer_dimensions)
        new_layer_index = random.randint(0, num_layers - 1)
        new_layer_size = random.randint(int(self.max_layer_size / 3), self.max_layer_size)

        self.layer_dimensions[new_layer_index] = new_layer_size



class GeneticEnvironment(gym.Env):
    """
    Custom environment implementing openAI gym interface
    """
    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects

        # Example when using discrete actions:
        N_DISCRETE_ACTIONS = 9
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input:
        #self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    
    def reset(self, get_complex_reward = True):
        print("Reset not implemented")
        
        return self.get_state()
    
    def get_reward(self):
        return 10.0 if self.answer == self.correct_answer else 0.0
    
    def step(self, action_index):
        print("Step not implemented")
        return observations, reward, done, info
    
    def get_state(self):
        print("Get state not implemented")
    
    def render(self, mode='human'):
        print("Render not implemented")
    
    def set_state(self, x, y):
        print("Set state not implemented")
    
    def close(self):
        print("Close not implemented")