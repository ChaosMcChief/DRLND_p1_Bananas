# DRLND_p1_Bananas
This repository is a possible solution to the first Project of the Deep reinforcement learning Nanodegree - the navigation task.

## Intro
The environment in this project is derived from the Unity Banana Collector environment and the solution is written for Pytorch. A quick description to the environment as well as to the state- and action-space can be found on the [Udacity Deep reinforcement learning github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

## What's in here?
All of the possible agents, which can be trained in this repo, base on the DDQN approach as described in this [paper](https://arxiv.org/pdf/1509.06461.pdf). On top of that you can compare two different improvements:
- Prioritized experience replay (as described [here](https://arxiv.org/pdf/1511.05952.pdf) and implemented [here](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb)
- Dueling Network Architecture (as described [here](https://arxiv.org/pdf/1511.06581.pdf))

## Instructions
This repo is tested in a Windows 64bit OS. If you use any different operating system, you have to set up the environment accordingly as describe in the already mentioned [udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) under "Getting Started". And don't forget to change the path to the environment folder in the Notebook.

The module `model.py` contains the definitions of the Pytorch-models and `dqn_agent.py` contains the definition of the agent as well as the experience memory.
In the jupyter-notebook `Navigation.ipynb` lies everything you need to start the environment and train the agent.

## ToDos
The implemented solution to the empty memory-problem regarding the prioritized experience replay (per) isn't an elegant one. Here you need to prepopulate the entire memory with experiences from random-actions before starting to train. A more elegant solution would be the use the standard memory at the beginning of training while populating the memory and then switching to the per.
