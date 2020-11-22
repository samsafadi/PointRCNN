
""" [summary]
"""

import torch
from sac_agent import SAC_Discrete
from pg import PG
from env import PointRCNNEnv
from memory import ReplayBuffer
import json
import argparse
from tqdm import tqdm


def train(agent, env, config, device, debug, replay_buffer=None):
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for reward-to-go weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []  # for measuring episode lengths
    s0 = env.reset()
    for t in tqdm(range(config['total_steps'])):
        # if t > start_steps:
        #     action = get_action(state.view(1,-1))
        # else:
        #     action = env.sample_act()
        # actually here we only have 1 step; we only excute 1 action
        a0 = agent.pick_action(s0, t)
        with torch.no_grad():
            s1, rew, done, _ = env.step(a0)

        if replay_buffer is None:
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)
        else:
            replay_buffer.push(s0.view(-1), a0.view(-1), reward.view(-1), s1.view(-1), done.view(-1))
        
        # actually this always true in our current setting
        if done:
            s0 = env.reset()

        # Update handling
        if t >= config['update_after'] and t % config['update_every'] == 0:
            # this seems to be important. that is: match the update steps to the update intervals
            # for j in range(config['update_every']):
            agent.update(replay_buffer, config['batch_size'])
    
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":

    config_path = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU {}'.format(torch.cuda.get_device_name(0)))
    else:
        args.device = torch.device('cpu')
        print('Running on CPU')

    config = load_config(config_path)
    debug = True

    # initialize the PointRCNNEnv and set it networks into eval mode
    env = PointRCNNEnv(config)

    # initialize the agent along with the networks inside it
    if config['policy'] == 'pg':
        agent = PG(config, env=env)
        replay_buffer = None
    elif config['policy'] == 'sac':
        agent = SAC_Discrete(config)
        replay_buffer = ReplayBuffer(config['replay_buffer_size'])

    train(agent, env, config, device, debug, replay_buffer)


