import _init_path
import os
import numpy as np
from env import PointRCNNEnv

my_env = PointRCNNEnv()

loader = my_env.test_loader
my_env.reset()
data = my_env.data

angle_map = data['angle_map']

# create random mask of zeros and ones
mask = np.random.randint(2, size=angle_map.shape[:3])
lidar_points = my_env._get_pts_from_mask(mask)

my_env._eval_data(lidar_points)
