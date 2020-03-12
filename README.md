# Behavioural and Cognitive Robotics (exercises)

## Exercise 1

You can find the implementation in the folder 'exercise_1'

## Exercise 2

### a)

Simple neural network controller for my Gym problem (CartPole-v0) implemented in '/exercise_1/exercise_2a.py'. In the end of each episode this program prints the total reward calculated as sum of the rewards over all steps. Since the parameters of the neural networks controller do not change among the episodes, the agent wiil not be able to balance the pole for many steps.

### b)

The evolutionary strategy for the neural network controller was implemented in '/exercise_1/exercise_2b.py'. I took the size of the population equals 10 and randomly initialized weights for each neural network using via normal distribution with average 0 and variance 0.1. Biases were set equal to 0.

The program is separated in two parts: training part and the post-evaluation one. During the first part the evolutionary strategy tries to find appropriate neural network controller. During the second stage I take the best neural network among the population and render the solution for the same number of episodes to show the robustness of this controller. 

I set the number of episodes equal to 10 and 200 steps in each one. Proposed method can solve the task, but not each time I run the program, because it is depends on the initial parameters (weights) of the neural networks in population. If we increase the number of episodes our controller will provide us better results. 

For this particular task 5 hidden neurons is enough and the increase of this parameter will not improve performance, but can slow down the search of a suitable neural network. The increase of the variance of the perturbation vector can affect the search in the same way.

## Exercise 3

![Results](./exercise_3/plotstat.png)