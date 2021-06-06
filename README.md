# multiagent-rl-kivy-
A custom TF-Agents reinforcement learning environment to accommodate simultaneous continuous actions from multiple agents.

The code features a custom Agent class, which creates agents with their own replay_buffers and networks. Similar to an example written by Dylan Cope (https://dylancope.github.io/Multiagent-RL-with-TFAgents/), but a continuous action space with a SAC network and importantly with the environment processing moves from multiple agents simultaneously. The code handles taking actions from each agent and creating a custom policy step which can be passed to the environment, whilst the agents use custom reward functions to manipulate their own timesteps.

The example code was written to facilitate agents learning with a Kivy app. This means I have to avoid while loops, but that I can make use of the Kivy Clock to schedule call backs e.g. when the game engine has a result, or to start the training loop.
