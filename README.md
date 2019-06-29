# gym_cartpole_dqn
**Pytorch Implementation of 2 types of DQN training: double DQN(DDQN) and vanilla DQN (DQN)**  
You can find explanations of the networks, for example, in "An Introduction to Deep Reinforcement Learning"  
by Vincent François-Lavet, Peter Henderson, Riashat Islam, Marc G. Bellemare and Joelle Pineau 5:  
DQN - section 4.3  
DDQN - section 4.4

**Training**  
Training is not very stable  
Successful training might be achieved in about 1-4 trials. So if training fails - please try again.  
Passes both cartpole v0 and cartpole v1 with both DQN and DDQN during training.  
Passing criterion for cartpole v0: average score on last 100 training episodes is above 195  
Passing criterion for cartpole v1: average score on last 100 training episodes is above 475  
**Plots for some successful training runs are given in the "plot" directory.  
Codes will be  updated if more stable/quick training is achieved.**

**Testing**  
<img src="https://media.giphy.com/media/dv7EKfnjCwiTvqosmG/giphy.gif">  
Performance at test time, provided successful training, with both DQN and DDQN:  
v0: always 200 (maximum limit of the environment is 200)  
v1: always 500 (maximum limit of the environment is 500)

**Input** to the network is 8x1 array [current state; current state - previous state], 
  where state is a 4x1 array returned by env.step(action)
