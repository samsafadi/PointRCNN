

"""policy gradient agent with advantage to sample discrete sampling action map
"""

import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from memory import ReplayBuffer  
from torch.distributions import Bernoulli
from UNet_model import UNet
from torch.autograd import Variable

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PG(object):
    def __init__(self, configs, env):
        self.configs = configs
        self.env = env
        self.actor = UNet()
        # make optimizer
        self.optimizer = Adam(actor.parameters(), lr=configs['lr'])


    def get_action(self, state, deterministic=False):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor(state)  # output size should be [B*H*W]
        action_probabilities = F.sigmoid(action_probabilities) # make sure the probs are in range [0,1]

        B, _, _ = action_probabilities.shape
        action_probabilities = action_probabilities.view(B, -1)
        # TODO leave this to future process; seems it will get the index
        max_probability_action = torch.argmax(action_probabilities, dim=-1)

        assert action_probabilities.size()[1, 2] == self.action_size, "Actor output the wrong size"
        if deterministic:
            # using deteministic policy during test time
            action = action_probabilities(action_probabilities>0.5).cpu()
        else:
            # using stochastic policy during traning time
            action_distribution = Bernoulli(action_probabilities)  # this creates a distribution to sample from
            action = action_distribution.sample().cpu()  # sample the discrete action and copy it to cpu

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)

        return action, action_probabilities, log_action_probabilities, max_probability_action
        

    def compute_loss(self, obs, act, rew):
        """make loss function whose gradient, for the right data, is policy gradient"""
        logp = self.get_action(obs).log_prob(act)
        act_baseline = self.get_action(obs, deterministic=True)
        _, rew_baseline, _, _ = self.env.step(obs, act_baseline)

        advantage = rew.to(device).float() - rew_baseline.to(device).float()

        loss = loss * Variable(advantage).expand_as(act)
        


        return loss


    def update(self, batch_obs, batch_acts, batch_rews):
        """take a single policy gradient update step for a batch"""
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  rew=torch.as_tensor(batch_rews, dtype=torch.int32),
                                  )
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss
        




        
    
