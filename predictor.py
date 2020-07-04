import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C

from stable_baselines.common.policies import FeedForwardPolicy, register_policy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras import backend


# Define custom policy
# Custom MLP policy of three layers of size 128 each
class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=1, n_lstm=64, reuse=False, **_kwargs):
        super(CustomLSTMPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        net_arch=[420, 'lstm', dict(vf=[128], pi=[128])],
        feature_extraction="mlp", **_kwargs)

# lstm = layers.LSTM(420, return_sequences=True)
# Define and Train the agent
class RLPredictorNet():
    def __init__(self, env):
        #self.model = A2C(MlpPolicy, env, verbose=1, gamma=0.5)
        env = DummyVecEnv([lambda: env])
        self.model = A2C(CustomLSTMPolicy, env, verbose=1, gamma=0.5)
    
    def train(self, total_timesteps=2000):
        self.model.learn(total_timesteps)


class SupervisedPredictorNet():
    def __init__(self, input_size, output_size):
        self.model = Sequential()
        self.model.add(LSTM(420, return_sequences=True))
        self.model.compile(loss='mse', optimizer="adam", metrics=["accuracy"])

    def train(self, x_train, y_train, batch_size=1, epochs=1, validation_split=0.1, save = False, verbose = False):
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
        print("k")
"""
class stable_baselines.common.policies.LstmPolicy(sess, ob_space, ac_space, 
n_env=1, 
n_batch, n
_lstm=256, reuse=False, layers=None, net_arch=None, act_fun=<MagicMock id='140641161579656'>, 
cnn_extractor=<function nature_cnn>, layer_norm=False, feature_extraction='cnn', **kwargs)

layers.Dense(num_units, activation="relu")
def seq2seq_model_builder(HIDDEN_DIM=300):
    
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model

model = seq2seq_model_builder(HIDDEN_DIM=300)
model.summary()

def LSTM_creator():
    # then create our final model
    model = keras.Sequential(shape=(5, 112, 112, 3), nbout=3)    # add the convnet with (5, 112, 112, 3) shape

    model.add(TimeDistributed(input_shape=shape))    # here, you can also use GRU or LSTM

    model.add(LSTM(64))    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

model.add(LSTM(data_dim, input_shape=(timesteps, data_dim)))
"""