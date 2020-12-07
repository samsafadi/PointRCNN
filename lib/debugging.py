import _init_path
import os
import numpy as np
from env import PointRCNNEnv

angle_map_path = '../data/KITTI/object/training/angle_map/'

my_env = PointRCNNEnv()

loader = my_env.test_loader
data = loader.dataset.__getitem__(0)

angle_map = data['angle_map']

print(angle_map[32, 512, :])

my_env.reset()
my_env._eval_data(depth_map)
