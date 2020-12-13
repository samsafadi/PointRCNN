

"""env file for interacting with data loader and PointRCNN detector
"""

import os
import sys
import logging
import torch
import json
import numpy as np
import re

from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.utils.bbox_transform import decode_bbox_target
from lib.utils import kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from torch.utils.data import DataLoader
from lib.net.point_rcnn import PointRCNN
import tools.train_utils.train_utils as train_utils
from tools.kitti_object_eval_python.eval import get_official_eval_result, eval_class, get_mAP
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
from tools.kitti_object_eval_python import kitti_common as kitti

HOME_DIR = os.path.join(os.getcwd(), '..')
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


def load_ckpt_based_on_cfg(config, model, logger):
    if config['ckpt'] is not None:
        train_utils.load_checkpoint(model, filename=config['ckpt'], logger=logger)

    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and config['rpn_ckpt'] is not None:
        load_part_ckpt(model, filename=config['rpn_ckpt'], logger=logger, total_keys=total_keys)

    if cfg.RCNN.ENABLED and config['rcnn_ckpt'] is not None:
        load_part_ckpt(model, filename=config['rcnn_ckpt'], logger=logger, total_keys=total_keys)


def create_dataloader(config, logger):
    mode = 'TEST' if config['test'] else 'EVAL'
    DATA_PATH = os.path.join('..', 'data')

    # create dataloader
    test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=config['random_select'],
                                rcnn_eval_roi_dir=config['rcnn_eval_roi_dir'],
                                rcnn_eval_feature_dir=config['rcnn_eval_feature_dir'],
                                classes=cfg.CLASSES,
                                logger=logger)

    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True,
                             num_workers=config['workers'], collate_fn=test_set.collate_batch)

    return test_loader


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


class PointRCNNEnv():
    def __init__(self, use_masked=True):
        super().__init__()
        np.random.seed(1024)
        
        # label path
        self.label_root = os.path.join(HOME_DIR, 'data/KITTI/object/training/label_2/')
        # load config
        config_path = os.path.join(HOME_DIR, 'tools/configs/pg.json')
        self.config = load_config(config_path)
        self.npoints = cfg.RPN.NUM_POINTS

        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')

        # create logger
        logger = create_logger(os.path.join(OUTPUT_DIR, 'log_pg.txt'))
        logger.info('**********************Start logging**********************')
        for key, val in self.config.items():
            logger.info("{:16} {}".format(key, val))
        save_config_to_file(cfg, logger=logger)

        # create PointRCNN dataloader & network
        self.test_loader = create_dataloader(self.config, logger)
        self.test_iter = iter(self.test_loader)
        self.model = PointRCNN(num_classes=self.test_loader.dataset.num_class, use_xyz=True, mode='TEST')

        self.use_masked = use_masked
        # load checkpoint
        load_ckpt_based_on_cfg(self.config, self.model, logger)

        # If want parallel
        # self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        self.model.eval()

        self.data = None

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

        # load the data sample at the reset step
        self.data = next(self.test_iter)
        RGB_Image = self.data['image']
        
        return RGB_Image

    def step(self, action):
        """step [Input the sampled map, output ]
        """

        # TODO: this is where we need to
        obs_pts = self._get_obs(action) # Here we output masked_pts as the obs 
        rew = self._get_reward(obs_pts) 

        # we set it as 1 step MDP so done is always true 
        done = True
        info = {}
        return obs_pts, rew, done, info 

    def _get_reward(self, obs):
        """step [Input the sampled point cloud, output the detection success]
        """
        batch_mAP = self._eval_data(masked_pts=obs)
        return batch_mAP
        

    def _get_obs(self, scanning_mask):
        """Here we set next obs as the sampled point cloud 
        """
        masked_pts = self._get_pts_from_mask(scanning_mask)
        return masked_pts

    def _get_pts_from_mask(self, scanning_mask):
        """ mask pts from 2d angular map
        Input: 
            :param scanning_mask: (B, H, W)
        Return:
            :param pts: (B, N, 4)
        """

        # load ang_depth_map from dir
        ang_depth_map = self.data['angle_map']

        # expand mask 2d->3d to enable broadcast
        mask = np.expand_dims(scanning_mask, axis=3)
        masked_ang_depth_map = [ang_depth_map[k] * mask[k] for k in range(self.config['batch_size'])]

        # masked_pts = masked_ang_depth_map.reshape((self.config['batch_size'], -1, 4))
        masked_pts_arr = [masked_pts[masked_pts[:, :, 0] > 0] for masked_pts in masked_ang_depth_map] # around ~(15000,4)

        adjusted_masked_pts = []
        for masked_pts in masked_pts_arr:
            if masked_pts.shape[0] <= self.npoints:
                padding = np.full((self.npoints - masked_pts.shape[0], 4), -1)
                adjusted_masked_pts.append(np.concatenate((masked_pts, padding), axis=0))
            else:
                adjusted_masked_pts.append(masked_pts[:self.npoints, :])

        masked_pts = np.array(adjusted_masked_pts)
        return masked_pts
    
    def render(self):
        """Placeholder for the rendering capacity
        """
        raise NotImplementedError

    def _eval_data(self, masked_pts=None):
        """eval data with sampled pts
        """
        with torch.no_grad():
            MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
            batch_size = self.config['batch_size']

            # get valid point (projected points should be in image)
            sample_id, pts_rect, pts_intensity, gt_boxes3d, npoints, labels = \
            self.data['sample_id'], self.data['pts_rect'], self.data['pts_intensity'], self.data['gt_boxes3d'], self.data['npoints'], self.data['label']

            cls_types = [[labels[k][i].cls_type for i in range(len(labels[k]))] for k in range(batch_size)]

            calib = [self.test_loader.dataset.get_calib(idx) for idx in sample_id]
            if self.use_masked:
                # use masked/sampled pts if True
                pts_rect = np.array([c.lidar_to_rect(masked_pts[k][:, 0:3]) for k, c in enumerate(calib)])
                pts_intensity = [masked_pts[k][:, 3] for k in range(batch_size)]
                npoints = masked_pts.shape[0]

            inputs = torch.from_numpy(pts_rect).cuda(non_blocking=True).float().view(self.config['batch_size'], -1, 3)
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True)
            input_data = {'pts_input': inputs}

            # model inference
            ret_dict = self.model(input_data)

            roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
            roi_boxes3d = ret_dict['rois']  # (B, M, 7)
            # seg_result = ret_dict['seg_result'].long()  # (B, N)

            rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
            rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

            norm_scores = torch.sigmoid(rcnn_cls)

            # remove low confidence scores
            thresh_mask = norm_scores > cfg.RCNN.SCORE_THRESH

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

            # select boxes (list of tensors)
            pred_boxes3d_selected = [pred_boxes3d[k][thresh_mask[k].view(-1)] for k in range(batch_size)]
            raw_scores_selected = [roi_scores_raw[k][thresh_mask[k].view(-1)] for k in range(batch_size)]
            norm_scores_selected = [norm_scores[k][thresh_mask[k].view(-1)] for k in range(batch_size)]

            # rotated NMS
            boxes_bev_selected = [kitti_utils.boxes3d_to_bev_torch(bboxes) for bboxes in pred_boxes3d_selected]
            keep_idx = [iou3d_utils.nms_gpu(boxes_bev_selected[k], raw_scores_selected[k], cfg.RCNN.NMS_THRESH).view(-1) for k in range(batch_size)]
            pred_boxes3d_selected = [pred_boxes3d_selected[k][keep_idx[k]] for k in range(batch_size)]
            scores_selected = [raw_scores_selected[k][keep_idx[k]] for k in range(batch_size)]
            norm_scores_selected = [norm_scores_selected[k][keep_idx[k]] for k in range(batch_size)]

            # want car gt_boxes
            keep_idx = [[i for i in range(len(cls_types[k])) if cls_types[k][i] == 'Car'] for k in range(batch_size)]
            gt_boxes3d_selected = [gt_boxes3d[k][keep_idx[k]] for k in range(batch_size)]

            # what if no boxes with cars?
            has_info = [k for k in range(batch_size) if len(keep_idx[k]) > 0]
            gt_boxes3d_selected = [gt_boxes3d_selected[x] for x in has_info]
            pred_boxes3d_selected = [pred_boxes3d_selected[x] for x in has_info]
            batch_size = len(has_info)
            if batch_size == 0:
                return None

            # Intersect over union
            iou3d = [iou3d_utils.boxes_iou3d_gpu(pred_boxes3d_selected[k], gt_boxes3d_selected[k]) for k in range(batch_size)]

            # get the max iou for each ground truth bounding box
            gt_max_iou = [torch.max(iou3d[k], dim=0)[0] for k in range(batch_size)]

            # get precision at each index (to get auc)
            precision_vals = []
            for k in range(batch_size):
                batch_iou = gt_max_iou[k]
                batch_precision = []
                num_correct = 0
                for i in range(len(batch_iou)):
                    if batch_iou[i] > 0.7:
                        num_correct += 1
                    batch_precision.append(num_correct / (i+1))

                precision_vals.append(batch_precision)
            
            aps = []
            for k in range(batch_size):
                batch_prec = precision_vals[k]
                ap = 0
                for i in range(len(batch_prec)):
                    ap += max(batch_prec[i:])

                aps.append(ap)
            
            num_gt_boxes = sum([len(gt_max_iou[k]) for k in range(batch_size)])
            
            return sum(aps) / num_gt_boxes