from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--sparse_data_dir", type=str, default=None, help="sparse data directory")
parser.add_argument("--dataset_name", type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
parser.add_argument("--match_num", type=int, default=0, help="number of sampled match pairs")
parser.add_argument("--skip_image", type=bool, default=False, help="do not generate images")
parser.add_argument("--generate_test", type=bool, default=False, help="generate test images")
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n, is_training):
    if is_training:
        frame_num = data_loader.num_train
        example = data_loader.get_train_example_with_idx(n)
    else:
        frame_num = data_loader.num_test
        example = data_loader.get_test_example_with_idx(n)

    if example == False:
        return
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, frame_num))

    # save image file
    if not args.skip_image:
        image_seq = concat_image_seq(example['image_seq'])
        dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
        scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))

    # save camera info
    if is_training:
        intrinsics = example['intrinsics']
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
        with open(dump_cam_file, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.\n' % (fx, cx, fy, cy))
            if 'pose' in example:
                poses = example['pose']
                for each_pose in poses:
                    f.write(','.join([str(num) for num in each_pose])+'\n')
            if 'match' in example:
                matches = example['match']
                for match in matches:
                    for i in range(match.shape[0]):
                        f.write(','.join([str(match[i,j]) for j in range(4)])+'\n')


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        args.sparse_data_dir,
                                        args.match_num,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       match_num=args.match_num,
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       match_num=args.match_num,
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        split='train',
                                        match_num=args.match_num,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)


    if args.generate_test:
        Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, is_training=False) for n in range(data_loader.num_test))
    else:
        Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, is_training=True) for n in range(data_loader.num_train))
        # Split into train/val
        np.random.seed(8964)
        subfolders = os.listdir(args.dump_root)
        with open(args.dump_root + 'train.txt', 'w') as tf:
            with open(args.dump_root + 'val.txt', 'w') as vf:
                for s in subfolders:
                    if not os.path.isdir(args.dump_root + '/%s' % s):
                        continue
                    imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                    frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                    for frame in frame_ids:
                        if np.random.random() < 0.1:
                            vf.write('%s %s\n' % (s, frame))
                        else:
                            tf.write('%s %s\n' % (s, frame))


if __name__ == '__main__':
    main()
