

"""env file for interacting with data loader and PointRCNN detector

TODO list
    - [ ] Enabling loading image in the KITTI dataloader, currently seems it only return pts 
    - [ ] connect this env with KITTI dataloader and detector 
    - [ ] add a script for projecting point cloud onto XY plane for sampling purpose
"""

import os
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from torch.utils.data import DataLoader
from lib.net.point_rcnn import PointRCNN
import numpy as np
import torch
import logging
import tools.train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import re


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


def load_ckpt_based_on_args(args, model, logger):
    if args.ckpt is not None:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)

    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        load_part_ckpt(model, filename=args.rcnn_ckpt, logger=logger, total_keys=total_keys)


def create_dataloader(args, logger):
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


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


class PointRCNNEnv():
    def __init__(self, args, logger):
        super().__init__()
        np.random.seed(1024)
        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)
        cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

        # cfg_file = 'cfgs/default.yaml'


        # define the logger here
        root_result_dir = os.path.join(root_result_dir, 'eval')
        # set epoch_id and output dir
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
        if args.test:
            root_result_dir = os.path.join(root_result_dir, 'test_mode')

        if args.extra_tag != 'default':
            root_result_dir = os.path.join(root_result_dir, args.extra_tag)
        os.makedirs(root_result_dir, exist_ok=True)

        log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
        logger = create_logger(log_file)
        logger.info('**********************Start logging**********************')
        for key, val in vars(args).items():
            logger.info("{:16} {}".format(key, val))
        save_config_to_file(cfg, logger=logger)

        # create PointRCNN dataloader & network
        self.test_loader = create_dataloader(logger)
        self.model = PointRCNN(num_classes=self.test_loader.dataset.num_class, use_xyz=True, mode='TEST')
        # load checkpoint
        load_ckpt_based_on_args(args, self.model, logger)

    def _batch_detector(self, batch_pts):
        """ Input a single or batch sample of point clouds, output prediction result
        """
        with torch.no_grad():
            self.model.eval()
            thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]



    def reset(self):
        """ reset env; here it is equivlent to load an image and a bin from the KITTI dataset. Set the image returned as s0

        data = {'sample_id': sample_id,
                'random_select': self.random_select,
                'pts_rect': pts_rect,
                'pts_intensity': pts_intensity,
                'gt_boxes3d': all_gt_boxes3d,
                'npoints': self.npoints,
                'image': image}
        """
        data = next(iter(self.test_loader))

        pts_intensity = data['pts_intensity']
        pts_rect = data['pts_rect']
        gt_boxes3d = data['gt_boxes3d']
        RGB_Image = data['image']
        
        return RGB_Image, pts_rect, pts_intensity, gt_boxes3d

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
