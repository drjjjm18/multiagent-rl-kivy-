# multiagent-rl-kivy-
A custom TF-Agents reinforcement learning environment to accommodate simultaneous continuous actions from multiple agents

The code features a custom Agent class, which creates agents with their own replay_buffers and networks. As inspired by https://dylancope.github.io/Multiagent-RL-with-TFAgents/, each agent ingests a timestep, which can be manipulated as appropriate for that agent with custom reward and observation functions. 

Unlike in the Dylan Cope example, I need the environment to process actions from both agents simultaneously. The code handles taking actions from each agent and creating a custom policy step which can be passed to the environment.
