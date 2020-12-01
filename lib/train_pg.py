import torch
import tqdm

from env import PointRCNNEnv

def train(agent, env, config, device, batch_size):
    """
    train pg model
    """
    # For logging
    batch_obs = []
    batch_acts = []
    batch_rets = []
    batch_lens = []

    # Starting state, reward, and trajectories
    state, done = env.reset(), False
    ep_rews = []

    while True:
        # Save state
        batch_obs.append(state.copy())

        # get the agent action for this state
        act, _, log_prob_action, _ = agent.get_action(state)

        with torch.no_grad():
            state, reward, done, _ = env.step(act)

        batch_acts.append(act)
        ep_rews.append(reward)

        if done:
            # Record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # reset episode-specific variables
            state, done, ep_rews = env.reset(), False, []

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    batch_loss = agent.update(batch_obs, batch_acts, batch_rets)
    return batch_loss, batch_rets, batch_lens


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('Running on CPU')
