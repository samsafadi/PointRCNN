"""sac agent to sample discrete sampling maps
"""

import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from memory import ReplayBuffer  
from torch.distributions import Bernoulli


class SAC_Discrete(object):
    def __init__(self, configs, env):
        self.configs = configs
        self.env = env
        # Define all the sub-networks

        # should be resenet like network, which output 1 value (q value) without softmax at the end. 
        self.q1 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.q2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
                            

        self.q1_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.q2_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)


    def get_action(self, state=None, global_step_number=0, eval=False):
        """
        Picks an action using one of three methods: 
            1) Randomly if we haven't passed a certain number of steps,
            2) Using the actor in evaluation mode if eval_ep is True  
            3) Using the actor in training mode if eval_ep is False.
        The difference between evaluation and training mode is that training mode does more exploration
         """
        if state is None:
            state = self.state
        if eval:
            action = self.get_action_from_actor(state, deterministic=True)
        elif global_step_number < self.configs["min_steps_before_learning"]:
            action = self.env.sample_action()
            print("Picking random action ", action)
        else:
            action = self.get_action_from_actor(state, deterministic=False)
        # if self.add_extra_noise:
        #     action += self.noise.sample()
        return action


    def get_action_from_actor(self, state, deterministic=False):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)  # output size should be [B*H*W]
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

        return action, (action_probabilities, log_action_probabilities), max_probability_action
    

    def update(self,replay_buffer, batch_size):
        pass


    def compute_loss_pi(self, data):
        """Set up function for computing SAC actor network loss"""
        o = data['obs']
        # pi, logp_pi = self.pi(o)
        a, (action_probabilities, log_action_probabilities), _ = self.get_action_from_actor(o)

        q1_pi = self.q1(o)
        q2_pi = self.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, alpha * logp_pi

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o)
        q2 = self.ac.q2(o)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # a2, logp_a2 = self.ac.pi(o2)
            a2, (p_a2, logp_a2), _ = self.get_action(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, self.alpha * logp_a2

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss