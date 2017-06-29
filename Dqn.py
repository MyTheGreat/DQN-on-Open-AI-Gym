import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random
from keras.optimizers import Adam


class Learner:
    def __init__(self, state_space_size, action_space_size, first_hidden_layer_size, second_hidden_layer_size):
        self.state_size = state_space_size
        self.action_size = action_space_size
        self.learning_rate = 0.001
        self.firstHidden = first_hidden_layer_size
        self.secondHidden = second_hidden_layer_size
        self.regressor = self._build_model()
        self.exploration = 1.
        self.exploration_decay = 0.99
        self.min_exploration = 0.01
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = 0.95

    def _build_model(self):
        regressor = Sequential()
        regressor.add(Dense(output_dim=self.firstHidden, input_dim=self.state_size, activation='relu'))
        regressor.add(Dense(output_dim=self.secondHidden, activation='relu'))
        regressor.add(Dense(output_dim=self.action_size, activation='linear'))
        regressor.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return regressor

    def act(self, state):
        if np.random.rand() <= self.exploration:
            # Better option for Mountain - Car
            # action = np.random.choice([0, 2])
            action = np.random.choice(range(self.action_size))
        else:
            action = np.argmax(self.regressor.predict(state), axis=1)[0]
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self):
        minibatch = random.sample(list(self.memory), self.batch_size)
        for state, action, reward, new_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.max(self.regressor.predict(new_state)[0])
            target_f = self.regressor.predict(state)
            target_f[0][action] = target
            self.regressor.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration > self.min_exploration:
            self.exploration *= self.exploration_decay
