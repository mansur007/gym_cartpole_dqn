# gym_cartpole_dqn
Pytorch implementation of DQN training - double DQN and vanilla DQN

This implementation is similar to the one from Pytorch tutorials:
  - tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
  - code: https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py  

But there is one important difference: 
  - input to the network is not a difference of 2 images, it is [current state; current state - previous state], 
  where state is a 4x1 array returned by env.step(action)
  
