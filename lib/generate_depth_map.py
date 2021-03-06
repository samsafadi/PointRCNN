# import scipy.misc as ssc
import imageio
import kitti_util
# import utils.calibration as calibration
import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# import kitti_util


def generate_dispariy_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    print('pts_2d', pts_2d.shape)
    # print('width', width)

    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)

    print('fov_inds', fov_inds.shape)
    print('pc_velo[1, 0]', pc_velo[1, 0])
    print('pts_2d[1, 1]', pts_2d[1, 1])

    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    print('valid_points', np.sum(fov_inds))
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    print('imgfov_pts_2d.shape', imgfov_pts_2d.shape)
    print('imgfov_pts_2d', imgfov_pts_2d[1])
    print('imgfov_pc_velo', imgfov_pc_velo[1])

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    return depth_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str,
                        default='/root/gdrive/My Drive/PointRCNN/data/KITTI/object/training')
    parser.add_argument('--split_file', type=str,
                        default='/root/gdrive/My Drive/PointRCNN/data/KITTI/ImageSets/train.txt')
    args = parser.parse_args()

    print('args.data_path', args.data_path)
    # assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/velodyne/'
    calib_dir = args.data_path + '/calib/'
    image_dir = args.data_path + '/image_2/'
    depth_dir = args.data_path + '/depth_map/'

    # assert os.path.isdir(lidar_dir)
    # assert os.path.isdir(calib_dir)
    # assert os.path.isdir(image_dir)

    if not os.path.isdir(depth_dir):
        os.makedirs(depth_dir)

    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
    lidar_files = sorted(lidar_files)

    assert os.path.isfile(args.split_file)
    with open(args.split_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    print('file_names', file_names[0])

    i = 0
    for fn in lidar_files:
        predix = fn[:-4]
        if predix not in file_names:
            continue
        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        # calib = kitti_util.Calibration(calib_file)
        calib = kitti_util.Calibration(calib_file)

        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn,
                            dtype=np.float32).reshape((-1, 4))[:, :3]
        print('lidar', lidar.shape)

        image_file = '{}/{}.png'.format(image_dir, predix)
        image = imageio.imread(image_file)
        print('image_shape', image.shape)

        height, width = image.shape[:2]

        depth_map = generate_dispariy_from_velo(lidar, height, width, calib)
        # print('depth_dir', depth_dir)
        print('depth_dir', depth_map.shape)
        print(depth_map)
        imageio.imwrite(depth_dir + '/' + predix+'.png', depth_map)
        np.save(depth_dir + '/' + predix, depth_map)
        print('Finish Depth Map {}'.format(predix))

        i = i+1
        if i>1:
            print(ccc)
