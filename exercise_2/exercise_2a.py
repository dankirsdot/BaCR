import gym
import time
import numpy as np

gym.logger.set_level(40)

env = gym.make('CartPole-v0')

pvariance = 0.1   # variance of initial parameters
nhiddens = 5      # number of hidden neurons

# the number of inputs and outputs depends on the problem
# we assume that observations consist of vectors of continuous value
# and that actions can be vectors of continuous values or discrete actions
ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

# initialize the training parameters randomly by using a gaussian distribution with average 0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0
W1 = np.random.randn(nhiddens, ninputs) * pvariance  # first layer
W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
b1 = np.zeros(shape=(nhiddens, 1))                   # bias first layer
b2 = np.zeros(shape=(noutputs, 1))                   # bias first layer

steps = 200   # number of steps
episodes = 10 # number of epochs
for e in range(episodes):
    reward_sum = 0
    observation = env.reset()
    for _ in range(steps):
        env.render()
        time.sleep(0.01)

        # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(ninputs,1)
        # compute the netinput of the first layer of neurons
        Z1 = np.dot(W1, observation) + b1
        # compute the activation of the first layer of neurons with the tanh function
        A1 = np.tanh(Z1)
        # compute the netinput of the second layer of neurons
        Z2 = np.dot(W2, A1) + b2
        # compute the activation of the second layer of neurons with the tanh function
        A2 = np.tanh(Z2)
        # if actions are discrete we select the action corresponding to the most activated unit
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)

        observation, reward, done, info = env.step(action)
        reward_sum = reward_sum + reward

    print("Reward during the episode {} equals {}".format(e + 1, reward_sum))

env.close()