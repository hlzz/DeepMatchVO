from __future__ import division
import numpy as np
from glob import glob
import os, sys
import scipy.misc

CURDIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(CURDIR, '..')))
sys.path.append(os.path.abspath(os.path.join(CURDIR, '...')))
from geo_utils import scale_intrinsics
from common_utils import load_match_func
from kitti_eval.pose_evaluation_utils import mat2euler

class kitti_odom_loader(object):
    def __init__(self,
                 dataset_dir,
                 sparse_data_dir=None,
                 match_num=0,
                 img_height=128,
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.sparse_data_dir = sparse_data_dir
        self.match_num = match_num
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]#, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        self.collect_test_frames()
        self.collect_train_frames()
        self.collect_train_poses()

    def collect_test_frames(self):
        self.test_frames = []
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.test_frames.append('%.2d %.6d' % (seq, n))
        self.num_test = len(self.test_frames)
        
    def collect_train_frames(self):
        self.train_frames = []
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.train_frames.append('%.2d %.6d' % (seq, n))
        self.num_train = len(self.train_frames)

    def collect_train_poses(self):
        self.train_poses = {}
        for seq in self.train_seqs:
            seq_poses = []
            pose_file = os.path.join(self.dataset_dir, 'poses', '%.2d.txt' % seq)
            with open(pose_file, 'r') as f:
                poses = f.readlines()
                for pose in poses:
                    pose_np = np.array(pose[:-1].split(' ')).astype(np.float32).reshape(3,4)    # kitti format is the inverse pose
                    rot = np.linalg.inv(pose_np[:,:3])
                    tran = -np.dot(rot, pose_np[:,3].transpose())
                    rz, ry, rx = mat2euler(rot)
                    seq_poses.append([rz, ry, rx] + tran.tolist())
            self.train_poses[seq] = seq_poses

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset+1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx, load_pose=False, load_matches=False):
        def _load_sparse_pose(drive, frame_id):
            sparse_data_file = os.path.join(self.sparse_data_dir, '%s/smea%s.txt' % (drive, frame_id))
            poses = []
            with open(sparse_data_file) as f:
                line = f.readline()
                abs_pose = [float(num) for num in line.split()]
                poses.append(abs_pose)
            half_offset = int((self.seq_length - 1)/2)
            frame_id_int = int(frame_id.lstrip('0'))
            for o in range(-half_offset, half_offset+1):
                if o == 0:
                    continue
                adj_frame_idx = frame_id_int + o
                adj_data_file = os.path.join(self.sparse_data_dir, '%s/smea%.6d.txt' % (drive, adj_frame_idx))
                with open(adj_data_file) as f:
                    line = f.readline()
                    adj_pose = [float(num) for num in line.split()]
                    poses.append(adj_pose)
            return poses
        
        def _load_gt_pose(drive, frame_id):
            half_offset = int((self.seq_length - 1)/2)
            drive_int = int(drive)
            frame_id_int = int(frame_id.lstrip('0'))
            poses = [self.train_poses[drive_int][frame_id_int]]
            for o in range(-half_offset, half_offset+1):
                if o == 0:
                    continue
                adj_frame_idx = frame_id_int + o
                poses.append(self.train_poses[drive_int][adj_frame_idx])
            return poses

        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics(tgt_drive, tgt_frame_id)
        intrinsics = scale_intrinsics(intrinsics, zoom_x, zoom_y)        
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        if load_pose:
            pose = _load_gt_pose(tgt_drive, tgt_frame_id)
            example['pose'] = pose
        if load_matches:
            sift_folder = os.path.join(self.dataset_dir, 'sequences', tgt_drive, 'sift')
            match_folder = os.path.join(self.dataset_dir, 'sequences', tgt_drive, 'match')
            matches = load_match_func(sift_folder, match_folder, tgt_frame_id, zoom_x, zoom_y, self.seq_length)
            example['match'] = matches
        return example

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx, load_pose=(
            self.sparse_data_dir != None), load_matches=(self.match_num != 0))
        return example
    

    def get_test_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.test_frames, tgt_idx):
            return False
        example = self.load_example(self.test_frames, tgt_idx, load_pose=False, load_matches=False)
        return example


    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, 'sequences', '%s/image_2/%s.png' % (drive, frame_id))
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir, 'sequences', '%s/calib.txt' % drive)
        proj_c2p, _ = self.read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics

    def read_calib_file(self, filepath, cid=2): # cid=2 and cid=3 has the same intrinsics
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data
        proj_c2p = parseLine(C[cid], shape=(3,4))
        proj_v2c = parseLine(C[-1], shape=(3,4))
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c
