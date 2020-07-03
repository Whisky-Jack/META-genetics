import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C

# Instantiate the env

# Define and Train the agent
class RLPredictorNet():
    def __init__(self, env):
        self.model = A2C(MlpPolicy, env, verbose=1, gamma=0.5)
    
    def train(self, total_timesteps=2000):
        self.model.learn(total_timesteps)

class SupervisedPredictorNet():
    def __init__(self):
        print("k")

    def train(self):
        print("k")