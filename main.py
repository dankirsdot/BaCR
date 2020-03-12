import gym
import time
import numpy as np

env = gym.make('CartPole-v0')
#env.reset()

pvariance = 0.1 # variance of initial parameters
ppvariance = 0.02 # variance of perturbations
nhiddens = 5 # number of hidden neurons

# the number of inputs and outputs depends on the problem
# we assume that observations consist of vectors of continuous value
# and that actions can be vectors of continuous values or discrete actions
ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n


networks = 10 # number of neural networks
epochs = 200 # number of epochs
# initialize the training parameters randomly by using a gaussian distribution with average 0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0
W1 = np.random.randn(nhiddens, ninputs, networks) * pvariance # first layer
W2 = np.random.randn(noutputs, nhiddens, networks) * pvariance # second layer
b1 = np.zeros(shape=(nhiddens, 1, networks)) # bias first layer
b2 = np.zeros(shape=(noutputs, 1, networks)) # bias first layer


for epoch in range(epochs):
    reward_sum = np.zeros(networks, dtype=float)
    for net in range(networks):
        observation = env.reset()
        for _ in range(1000):
            #time.sleep(0.1)
            # convert the observation array into a matrix with 1 column and ninputs rows
            observation.resize(ninputs,1)
            # compute the netinput of the first layer of neurons
            Z1 = np.dot(W1[:, :, net], observation) + b1[:, :, net]
            # compute the activation of the first layer of neurons with the tanh function
            A1 = np.tanh(Z1)
            # compute the netinput of the second layer of neurons
            Z2 = np.dot(W2[:, :, net], A1) + b2[:, :, net]
            # compute the activation of the second layer of neurons with the tanh function
            A2 = np.tanh(Z2)
            # if actions are discrete we select the action corresponding to the most activated unit
            if (isinstance(env.action_space, gym.spaces.box.Box)):
                action = A2
            else:
                action = np.argmax(A2)

            if net == 0:
                env.render()

            observation, reward, done, info = env.step(action)
            reward_sum[net] = reward_sum[net] + reward
            
    best = reward_sum.argsort()[-5:][::-1]
    worst = reward_sum.argsort()[:5][::-1]
    print('Reward', reward_sum)
    print('Best', best)

    W1[:, :, worst] = W1[:, :, best] + np.random.randn(nhiddens, ninputs, networks//2) * ppvariance
    W2[:, :, worst] = W2[:, :, best] + np.random.randn(noutputs, nhiddens, networks//2) * ppvariance
    b1[:, :, worst] = b1[:, :, best] + np.random.randn(nhiddens, 1, networks//2) * ppvariance
    b2[:, :, worst] = b2[:, :, best] + np.random.randn(noutputs, 1, networks//2) * ppvariance

env.close()