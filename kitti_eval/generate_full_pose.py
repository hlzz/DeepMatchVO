from __future__ import division
import os, sys
import math
import scipy.misc
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import mat2euler, quat2mat, pose_vec2mat

CURDIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(CURDIR, '.')))
sys.path.append(os.path.abspath(os.path.join(CURDIR, '..')))
sys.path.append(os.path.abspath(os.path.join(CURDIR, '...')))
from common_utils import is_valid_sample

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="path to kitti odometry dataset")
parser.add_argument("--snippet_dir", type=str, help="path to segmental pose dataset")
parser.add_argument("--output_dir",  type=str, help="path to output pose snippets")
parser.add_argument("--seq_id",      type=int, default=9, help="sequence id to generate groundtruth pose snippets")
parser.add_argument("--seq_length",  type=int, default=3, help="sequence length of pose snippets")
args = parser.parse_args()


def decode_pose(one_pose, first_pose):
    """return the pose mat in kitti ground-truth and the pose needed for next segment"""
    _, tx, ty, tz, qx, qy, qz, qw = one_pose.strip().split()
    rot = quat2mat([float(qw), float(qx), float(qy), float(qz)])
    rz, ry, rx = mat2euler(rot)
    pose_inv = pose_vec2mat([tx, ty, tz, rx, ry, rz], True).astype(np.float32)
    pose = np.dot(np.linalg.inv(pose_inv), first_pose)

    rot = pose[:3,:3].transpose().astype(float)
    trans = -np.dot(rot, pose[:3,3].transpose())
    Tmat = np.concatenate((rot, trans.reshape(3,1)), axis=1)
    return Tmat, pose


def main():
    pose_gt_dir = args.dataset_dir + 'poses/'
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.dataset_dir + 'sequences/%.2d/times.txt' % args.seq_id, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    segment_pose_files = glob(args.snippet_dir+'/*.txt')
    segment_num = len(segment_pose_files)
    all_poses = []
    first_abs_pose = np.eye(4, dtype=float)
    anchor_poses = []
    for file_num in range(segment_num):
        filename = os.path.join(args.snippet_dir, '%.6d.txt' % file_num)
        with open(filename, 'r') as f:
            lines = f.readlines()
        if file_num == 0:
            for one_pose in lines:
                Tmat, anchor_pose = decode_pose(one_pose, first_abs_pose)
                all_poses.append(Tmat)
                anchor_poses.append(anchor_pose)
        else:
            first_pose = anchor_poses[file_num]
            # first adjust previous added frames
            for i in range(1, len(lines)-1):
                # TODO(tianwei): motion average currently only works with seq_length=3, this average is just arithmic
                Tmat, anchor_pose = decode_pose(lines[i], first_pose)
                all_poses[file_num+i] = (all_poses[file_num+i] + Tmat)/2
                anchor_poses[file_num+i] = (anchor_poses[file_num+i] + anchor_pose)/2
            Tmat, anchor_pose = decode_pose(lines[-1], first_pose)
            all_poses.append(Tmat)
            anchor_poses.append(anchor_pose)
            
    print(len(all_poses), 'total frames')
    with open(os.path.join(args.output_dir, ('%.2d' % args.seq_id) + '_full.txt'), 'w') as f:
        for pose in all_poses:
            pose = pose.reshape((12,1))
            pose_str = str(float(pose[0]))
            for i in range(1,12):
                pose_str = pose_str+' '+str(float(pose[i]))
            f.write(pose_str+'\n')


if __name__ == '__main__':
    main()