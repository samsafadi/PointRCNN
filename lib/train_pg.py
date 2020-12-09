import torch
import tqdm
import json
import sys

sys.path.append('../')

import argparse
from lib.env import PointRCNNEnv
from lib.pg_agent import PG

def train(agent, env, config, device):
    """
    train pg model
    """
    # For logging
    batch_obs = []
    batch_acts = []
    batch_rets = []
    batch_lens = []

    # Starting state, reward, and trajectories
    for _ in tqdm.tqdm(range(len(env.test_loader))):
        # loads a new state (image)
        state, done = env.reset(), False

        ep_rews = []

        while True:
            # Save state
            batch_obs.append(state.copy())

            # get the agent action for this state
            act, prob_action, log_prob_action, _ = agent.get_action(state)

            # get reward
            obs, batch_reward, done, _ = env.step(act)

            # calculates loss against baseline and steps optimizer
            batch_loss = agent.update(batch_obs, batch_acts, batch_rets)

            batch_obs.append(obs)
            batch_acts.append(act)
            ep_rews.append(batch_reward)

            if done:
                # Record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # reset episode-specific variables
                state, done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > config['batchsize']:
                    break

    return batch_loss, batch_rets, batch_lens


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    
    parser.add_argument("--eval_mode", type=str, default='rpn', required=True, help="specify the evaluation mode")
    parser.add_argument("--num_epochs", type=int, default=10, required=False, help="specify the number of epochs")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('Running on CPU')

    config_path = '/pg.json'
    config = load_config(config_path)
    debug = True

    # initialize the PointRCNNEnv and set it networks into eval mode
    env = PointRCNNEnv()

    # initialize the agent along with the networks inside it
    agent = PG(config, env=env)

    for i in range(epochs):
        train(agent, env, config, device)
