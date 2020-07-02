import gym
from gym import spaces
from gym.envs.classic_control import rendering

# DEFINE ENVIRONMENT
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