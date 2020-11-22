

"""env file for interacting with data loader and PointRCNN detector
"""


class PointRCNNEnv(object):
    def __init__(self):
        super().__init__()


    def reset(self):
        """ reset env; here it is equivlent to load an image and a bin from the KITTI dataset. Set the image as s0
        """
        pass

    def step(self, action, obs=None):
        """step [Input the sampled map, output ]
        """
        return obs, rew, done, info 


    def get_reward(self, obs):
        """step [Input the sampled map, output ]
        """
        pass

    
    def render(self):
        """Placeholder for the rendering capacity
        """
        raise NotImplementedError
