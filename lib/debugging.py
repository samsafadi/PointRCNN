import _init_path
from env import PointRCNNEnv

my_env = PointRCNNEnv()

loader = my_env.test_loader
data = loader.dataset.__getitem__(0)
print(data)
