"""sac agent to sample discrete sampling maps
"""


import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from memory import ReplayBuffer  

class SAC_discrete(object):
    def __init__(self, config):
        self.hyperparameters = config.hyperparameters

        # Define all the sub-networks

        # should be resenet like network, which output 1 value (q value) without softmax at the end. 
        self.q1 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.q2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)

        self.critic_optimizer = torch.optim.Adam(self.q1.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.q2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
                            

        self.q1_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.q2_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")


        # setup replay buffer
        # the replay buffer might be changed into the datalodar mode in the future in order to ease the burden for memory 
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
    
    def get_action(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state) # size should be [B*H*W]
        B, _, _ = action_probabilities.shape
        action_probabilities = action_probabilities.view(B, -1)
        # TODO leave this to future process; seems it will get the index
        max_probability_action = torch.argmax(action_probabilities, dim=-1)

        assert action_probabilities.size()[1,2] == self.action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
        action = action_distribution.sample().cpu()  # sample the discrete action

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)

        return action, (action_probabilities, log_action_probabilities), max_probability_action
    


    def update(self, )



    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        # pi, logp_pi = self.pi(o)
        action, (action_probabilities, log_action_probabilities), _ = self.get_action(state_batch)

        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, alpha * logp_pi
