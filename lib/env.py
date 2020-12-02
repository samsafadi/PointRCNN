

"""env file for interacting with data loader and PointRCNN detector

TODO list
    - [ ] Enabling loading image in the KITTI dataloader, currently seems it only return pts 
    - [ ] connect this env with KITTI dataloader and detector 
    - [ ] add a script for projecting point cloud onto XY plane for sampling purpose
"""

import os
import logging
import torch
import numpy as np

from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.utils.bbox_transform import decode_bbox_target
from torch.utils.data import DataLoader
from lib.net.point_rcnn import PointRCNN
import numpy as np
import torch
import logging
import tools.train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import re

OUTPUT_DIR = '../output/pg_log/'

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


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


class PointRCNNEnv():
    def __init__(self):
        super().__init__()
        np.random.seed(1024)
        
        # create logger
        logger = create_logger(os.path.join(OUTPUT_DIR, 'log_pg.txt'))
        logger.info('**********************Start logging**********************')
        for key, val in vars(args).items():
            logger.info("{:16} {}".format(key, val))
        save_config_to_file(cfg, logger=logger)
        
        # create PointRCNN dataloader & network
        self.test_loader = create_dataloader(logger)
        self.model = PointRCNN(num_classes=self.test_loader.dataset.num_class, use_xyz=True, mode='TEST')

        # load checkpoint
        load_ckpt_based_on_args(args, self.model, logger)

    # def _batch_detector(self, batch_pts):
    #     """ Input a single or batch sample of point clouds, output prediction result
    #     """
    #     with torch.no_grad():
    #         self.model.eval()
    #         thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]


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
        # TODO: this is where we need to
        obs = self._get_obs(scanning_mask=action)
        rew = self._get_reward(obs)

        # we set it as 1 step MDP so done is always true 
        done = True
        info = {}
        return obs, rew, done, info 

    def _get_reward(self, obs):
        """step [Input the sampled point cloud, output the detection success]
        """
        # Guangyuan: batchsize should be 1, right?
        batch_mAP = self._eval_data(obs)

        return batch_mAP
        

    def _get_obs(self, data):
        """Here we set next obs as the sampled point cloud 
        """
        data['npoints'] = self._get_pts_from_mask()
        obs = data['npoints']

        return obs

    def _get_pts_from_mask(self, masks):
        
        pass 
    
    def render(self):
        """Placeholder for the rendering capacity
        """
        raise NotImplementedError

    def _eval_data(self, data):
        """eval data with RCNN model
        """
        sample_id, pts_rect, pts_intensity, gt_boxes3d, npoints = \
            data['sample_id'], data['pts_rect'], data['pts_intensity'], data['gt_boxes3d'], data['npoints']

        inputs = torch.from_numpy(pts_rect).cuda(non_blocking=True).float()
        input_data = {'pts_input': inputs}

        # model inference
        ret_dict = self.model(input_data)

        roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].long()  # (B, N)

        # set batch size to one for now
        batch_size = 1

        rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

        # bounding box regression
        anchor_size = MEAN_SIZE

        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
                                          loc_scope=cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        # Intersect over union
        iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[0], gt_boxes3d)
        gt_max_iou, _ = iou3d.max(dim=0)

        # Recall is how many of the gt boxes were predicted
        recalled_num = (gt_max_iou > 0.7).sum().item()
        total_boxes = len(gt_boxes3d)

        # Return ratio unless total boxes is zero
        if total_boxes == 0:
            return None
        else:
            return recalled_num / total_boxes
