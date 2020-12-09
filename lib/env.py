

"""env file for interacting with data loader and PointRCNN detector

TODO list
    - [ ] Enabling loading image in the KITTI dataloader, currently seems it only return pts 
    - [ ] connect this env with KITTI dataloader and detector 
    - [ ] add a script for projecting point cloud onto XY plane for sampling purpose
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
from tools.kitti_object_eval_python.eval import get_official_eval_result, eval_class
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
        self.model = PointRCNN(num_classes=self.test_loader.dataset.num_class, use_xyz=True, mode='TEST')
        self.model.cuda()
        self.model.eval()

        self.use_masked = use_masked
        # load checkpoint
        load_ckpt_based_on_cfg(self.config, self.model, logger)

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
        self.data = next(iter(self.test_loader))
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
        return obs, rew, done, info 

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
            :param mask: (H, W)
            :param pts_intensity: (N, 1)
        Return:
            :param pts: (N, 4)
        """

        # load ang_depth_map from dir
        ang_depth_map = np.squeeze(self.data['angle_map'], axis=0)
        # ang_depth_map = np.load(os.path.join(
        #     self.config.angle_map_dir, "{:06d}.npy".format(self.data['sample_id'])))  # (H,W,4)

        # expand mask 2d->3d to enable broadcast
        mask = np.expand_dims(scanning_mask, axis=2)
        masked_ang_depth_map = ang_depth_map * mask

        masked_pts = masked_ang_depth_map.reshape((-1, 4))
        # masked_pts = masked_pts[masked_pts[:, 0] != -1.0) & (masked_pts[:, 0] != 0)]  # around ~(15000,4)
        masked_pts = masked_pts[masked_pts[:, 0] > 0]  # around ~(15000,4)

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

            # get valid point (projected points should be in image)
            sample_id, pts_rect, pts_intensity, gt_boxes3d, npoints = \
            self.data['sample_id'], self.data['pts_rect'], self.data['pts_intensity'], self.data['gt_boxes3d'], self.data['npoints']

            # TODO try to access this with calib function
            calib = self.test_loader.dataset.get_calib(sample_id)
            if self.use_masked:
                # use masked/sampled pts if True
                pts_rect = calib.lidar_to_rect(masked_pts[:, 0:3])
                pts_intensity = masked_pts[:, 3]
                npoints = masked_pts.shape[0]

            inputs = torch.from_numpy(pts_rect).cuda(non_blocking=True).float().view(1, -1, 3)
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True)
            input_data = {'pts_input': inputs}

            # model inference
            ret_dict = self.model(input_data)

            roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
            roi_boxes3d = ret_dict['rois']  # (B, M, 7)
            seg_result = ret_dict['seg_result'].long()  # (B, N)

            # say batch size is one for now
            pred_score = roi_scores_raw[0]

            # print(pred_score)

            # set batch size to one for now
            batch_size = 1

            rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
            rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

            # print(rcnn_cls)
            norm_scores = torch.sigmoid(rcnn_cls)
            print(norm_scores[norm_scores > 0.7])

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
            iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[0], torch.squeeze(gt_boxes3d))
            gt_max_iou, _ = iou3d.max(dim=0)

            # Recall is how many of the gt boxes were predicted
            recalled_num = (gt_max_iou > 0.7).sum().item()
            total_boxes = len(gt_boxes3d)
            
            """
            Annotations - how kitti eval interprets the information from kitti_common.py
            annotations.update({                                                                                                                                                                                                                                                                                                                
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': []
            })

            How it saves to the file in eval_rcnn.py
            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)

            How it reads from it into annotations
            annotations['name'] = np.array([x[0] for x in content])
            annotations['truncated'] = np.array([float(x[1]) for x in content])
            annotations['occluded'] = np.array([int(x[2]) for x in content])
            annotations['alpha'] = np.array([float(x[3]) for x in content])
            annotations['bbox'] = np.array(
                [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
            # dimensions will convert hwl format to standard lhw(camera) format.
            annotations['dimensions'] = np.array(
                [[float(info) for info in x[8:11]] for x in content]).reshape(
                    -1, 3)[:, [2, 0, 1]]
            annotations['location'] = np.array(
                [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
            annotations['rotation_y'] = np.array(
                [float(x[14]) for x in content]).reshape(-1)
            if len(content) != 0 and len(content[0]) == 16:  # have score
                annotations['score'] = np.array([float(x[15]) for x in content])
            else:
                annotations['score'] = np.zeros([len(annotations['bbox'])])

            NOTE: TO GET ALPHA
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            OCCLUDED AND TRUNCATED SEEM TO ALWAYS BE -1

            NOTE: TO GET IMAGE BOXES
            corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
            img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
            """

            # convert to np
            pred_boxes3d_np = np.squeeze(pred_boxes3d.cpu().numpy(), axis=0)
            gt_boxes3d_np = np.squeeze(gt_boxes3d.cpu().numpy(), axis=0)

            pred_annos = {}
            pred_annos.update({
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': []
            })

            pred_annos['name'] = np.array([cfg.CLASSES for _ in range(len(pred_boxes3d_np))])
            pred_annos['truncated'] = np.array([-1 for _ in range(len(pred_boxes3d_np))])
            pred_annos['occluded'] = np.array([-1 for _ in range(len(pred_boxes3d_np))])

            # Get image boxes
            corners3d = kitti_utils.boxes3d_to_corners3d(pred_boxes3d_np)
            img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
            pred_annos['bbox'] = img_boxes

            x, z, ry = pred_boxes3d_np[:, 0], pred_boxes3d_np[:, 2], pred_boxes3d_np[:, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry
            
            pred_annos['alpha'] = alpha
            # reorder to convert hwl format to standard lhw(camera) format.
            pred_annos['dimensions'] = pred_boxes3d_np[:, 3:6][:, [2, 0, 1]]
            pred_annos['location'] = pred_boxes3d_np[:, 0:3]
            pred_annos['rotation_y'] = pred_boxes3d_np[:, 6]
            pred_annos['score'] = pred_score.cpu().numpy()

            pred_annos = [pred_annos]
        
            # for k, bbox3d in enumerate(pred_boxes3d_np):
            #     bbox3d = np.expand_dims(bbox3d, axis=0)

            #     anno['name'].append(cfg.CLASSES)
            #     anno['truncated'].append(-1)
            #     anno['occluded'].append(-1)

            #     # Get image boxes
            #     corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
            #     img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
            #     anno['bbox'].append(img_boxes)

            #     x, z, ry = bbox3d[0, 0], bbox3d[0, 2], bbox3d[0, 6]
            #     beta = np.arctan2(z, x)
            #     alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            #     anno['alpha'] = np.array([alpha])
            #     # dimensions will convert hwl format to standard lhw(camera) format.
            #     anno['dimensions'] = np.expand_dims(np.array([bbox3d[0, 5], bbox3d[0, 3], bbox3d[0, 4]]), axis=0)
            #     anno['location'] = np.expand_dims(bbox3d[0, 0:3], axis=0)
            #     anno['rotation_y'] = np.expand_dims(np.array([bbox3d[0, 6]]), axis=0)
            #     anno['score'] = np.array([pred_score[k]])
            #     pred_annos.append(anno)

            # label_annos = []
            # for k, bbox3d in enumerate(gt_boxes3d_np):
            #     anno = {}
            #     anno.update({
            #         'name': [],
            #         'truncated': [],
            #         'occluded': [],
            #         'alpha': [],
            #         'bbox': [],
            #         'dimensions': [],
            #         'location': [],
            #         'rotation_y': []
            #     })

            #     bbox3d = np.expand_dims(bbox3d, axis=0)

            #     anno['name'] = np.array(cfg.CLASSES)
            #     anno['truncated'], anno['occluded'] = np.array([-1]), np.array([-1])

            #     # Get image boxes
            #     calib = self.test_loader.dataset.get_calib(sample_id)
            #     corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
            #     img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
            #     anno['bbox'] = np.expand_dims(img_boxes[0, 0:3], axis=0)

            #     x, z, ry = bbox3d[0, 0], bbox3d[0, 2], bbox3d[0, 6]
            #     beta = np.arctan2(z, x)
            #     alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            #     anno['alpha'] = np.array([alpha])
            #     # dimensions will convert hwl format to standard lhw(camera) format.
            #     anno['dimensions'] = np.expand_dims(np.array([bbox3d[0, 5], bbox3d[0, 3], bbox3d[0, 4]]), axis=0)
            #     anno['location'] = np.expand_dims(bbox3d[0, 0:3], axis=0)
            #     anno['rotation_y'] = np.expand_dims(np.array([bbox3d[0, 6]]), axis=0)
            #     label_annos.append(anno)

            label_anno = kitti.get_label_anno(os.path.join(self.label_root, '%06d.txt' % sample_id))
            label_annos = [label_anno]

            # 0 means car
            current_classes = [0]

            overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                                     0.5], [0.7, 0.5, 0.5, 0.7, 0.5],
                                    [0.7, 0.5, 0.5, 0.7, 0.5]])
            overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7,
                                     0.5], [0.5, 0.25, 0.25, 0.5, 0.25],
                                    [0.5, 0.25, 0.25, 0.5, 0.25]])
            min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
            class_to_name = {
                0: 'Car',
                1: 'Pedestrian',
                2: 'Cyclist',
                3: 'Van',
                4: 'Person_sitting',
            }
            name_to_class = {v: n for n, v in class_to_name.items()}
            if not isinstance(current_classes, (list, tuple)):
                current_classes = [current_classes]
            current_classes_int = []
            for curcls in current_classes:
                if isinstance(curcls, str):
                    current_classes_int.append(name_to_class[curcls])
                else:
                    current_classes_int.append(curcls)
            current_classes = current_classes_int
            min_overlaps = min_overlaps[:, :, current_classes]

            # return get_official_eval_result(label_annos, pred_annos, cfg.CLASSES)
            # print(label_annos)
            # print(pred_annos)
            difficultys = [1]
            ret = eval_class(label_annos, pred_annos, current_classes, difficultys, 0, min_overlaps, num_parts=1)
            print(ret)
