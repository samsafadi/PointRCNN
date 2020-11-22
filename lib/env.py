

"""env file for interacting with data loader and PointRCNN detector

TODO list
    - [ ] connect this env with KITTI dataloader and detector 
    - [ ] add a script for projecting point cloud onto XY plane for sampling purpose
"""

import os
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.net.point_rcnn import PointRCNN


def create_dataloader(logger):
    mode = 'TEST' if args.test else 'EVAL'
    DATA_PATH = os.path.join('..', 'data')

    # create dataloader
    test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=args.random_select,
                                rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir=args.rcnn_eval_feature_dir,
                                classes=cfg.CLASSES,
                                logger=logger)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=test_set.collate_batch)

    return test_loader


class PointRCNNEnv(object):
    def __init__(self):
        super().__init__()
        np.random.seed(1024)
        # create PointRCNN dataloader & network
        self.test_loader = create_dataloader(logger)
        self.model = PointRCNN(num_classes=self.test_loader.dataset.num_class, use_xyz=True, mode='TEST')
        # load checkpoint
        load_ckpt_based_on_args(self.model, logger)

    def _batch_detector(self, batch_pts):
        """ Input a single or batch sample of point clouds, output prediction result
        """
        with torch.no_grad():
            self.model.eval()
            thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]



    def reset(self):
        """ reset env; here it is equivlent to load an image and a bin from the KITTI dataset. Set the image returned as s0
        """
        return RGB_Image 

    def step(self, action, obs=None):
        """step [Input the sampled map, output ]
        """
        return obs, rew, done, info 

    def _get_reward(self, obs):
        """step [Input the sampled point cloud, output the detection success]
        """
        batch_mAP = self.step()
        

    def _get_obs(self):
        """Here we set next obs as the sampled point cloud 
        """
        return sampled_pts
    
    def render(self):
        """Placeholder for the rendering capacity
        """
        raise NotImplementedError
