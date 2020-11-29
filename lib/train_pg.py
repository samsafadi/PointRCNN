import torch

from env import PointRCNNEnv

def train(agent, env, config, device):
    """
    train pg model
    """
    return

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('Running on CPU')
