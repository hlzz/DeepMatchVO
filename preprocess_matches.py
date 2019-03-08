#!/usr/bin/python
# Tianwei Shen, HKUST, 2019.
# Copyright reserved.
# This file is an example to parse the feature and matching file,
# in accord with our internal format.

from __future__ import print_function

import os
import sys
import glob 
import numpy as np
import math
from struct import unpack
from PIL import Image, ImageDraw

# REPLACE these paths with yours
sift_list_path = '/home/tianwei/Data/kitti/odometry/dataset/odometry/sequences/00/sift_list.txt'
match_folder = '/home/tianwei/Data/kitti/odometry/dataset/odometry/sequences/00/match'

def read_feature_repo(file_path):
    """Read feature file (*.sift)."""
    with open(file_path, 'rb') as fin:
        data = fin.read()

    head_length = 20
    head = data[0:head_length]
    feature_name, _, num_features, loc_dim, des_dim = unpack('5i', head)
    keypts_length = loc_dim * num_features * 4

    if feature_name == ord('S') + (ord('I') << 8) + (ord('F') << 16) + (ord('T') << 24):
        print(Notify.INFO, 'Reading SIFT file',
              file_path, '#', num_features, Notify.ENDC)
        desc_length = des_dim * num_features
        desc_type = 'B'
    elif feature_name == 21384864:  # L2Net
        print(Notify.INFO, 'Reading L2NET file',
              file_path, '#', num_features, Notify.ENDC)
    else:
        print(Notify.FAIL, 'Unknown feature type.', Notify.ENDC)
        desc_length = des_dim * num_features * 4
        desc_type = 'f'

    keypts_data = data[head_length: head_length + keypts_length]
    keypts = np.array(unpack('f' * loc_dim * num_features, keypts_data))
    keypts = np.reshape(keypts, (num_features, loc_dim))

    desc_data = data[head_length +
                     keypts_length: head_length + keypts_length + desc_length]
    desc = np.array(unpack(
        desc_type * des_dim * num_features, desc_data))
    desc = np.reshape(desc, (num_features, des_dim))
    return keypts, desc


def read_match_repo(mat_file):
    """Read .mat file and read matches
    
    Arguments:
        mat_file {str} -- .mat file
    
    Returns:
        A list of tuples with each of format (second_sift_name (without .sift suffix), 
        match_num (putative, hinlier, finlier), homograph matrix, fundamental matrix,
        match pairs (list of (feat1, feat2, flag)))
    """

    match_ret = []
    with open(mat_file, 'rb') as fin:
        data = fin.read()
        if len(data) == 0:
            return match_ret
        file_end = len(data)

        end = 0
        while True:
            # read filename length
            length_bytes = 4
            length = data[end:end+length_bytes]
            length = unpack('i', length)[0]
            end += length_bytes

            # read filename
            filename_bytes = length
            filename = data[end:end+filename_bytes]
            filename = unpack('c' * length, filename)
            sift_name2 = os.path.splitext(''.join(filename))[0]
            end += filename_bytes

            # read match number (putative, hinlier, finlier)
            match_num_bytes = 4 * 3
            match_num = data[end:end+match_num_bytes]
            match_num = unpack('3i', match_num)
            end += match_num_bytes

            # read homograph (3x3) and fundamental matrix (3x3)
            mat_bytes = 8 * 18
            mat = data[end:end+mat_bytes]
            mat = unpack('18d', mat)
            hmat = mat[:9]
            fmat = mat[9:]
            hmat = np.matrix([hmat[:3],hmat[3:6],hmat[6:9]], dtype=np.float32)
            fmat = np.matrix([fmat[:3],fmat[3:6],fmat[6:9]], dtype=np.float32)
            end += mat_bytes

            # read actual match (sift feature index pairs)
            struct_bytes = 12 * match_num[0]
            struct = data[end:end+struct_bytes]
            struct = unpack(match_num[0] * '3i', struct)
            struct = np.reshape(struct, (-1, 3))
            end += struct_bytes

            match_ret.append((sift_name2, match_num, hmat, fmat, struct))
            if end == file_end:
                break
    return match_ret


def get_inlier_image_coords(sift_keys1, sift_keys2, feature_matches, type='f'):
    """Get inlier matches in image coordinates.
    
    Arguments:
        sift_keys1 {list of keys (x, y, color, scale, orientation)} -- first sift keys 
        sift_keys2 {list of keys} -- second sift keys
        feature_matches {(first, second, flag)} -- sift key index pairs and flags
    
    Keyword Arguments:
        type {str} -- inlier type ('f' for fudamental matrix and 'h' for homography) (default: {'f'})
    
    Returns:
        list -- list of (x1, y1, x2, y2)
    """

    image_matches = []
    if type == 'f':
        inlier_type = 2
    elif type == 'h':
        inlier_type = 1
    else:
        print('Unknown inlier type, should be "f" or "h"')
        exit(-1)

    for i in range(feature_matches.shape[0]):
        if (feature_matches[i, 2] == inlier_type or feature_matches[i, 2] == 3):
            index1 = feature_matches[i, 0]
            index2 = feature_matches[i, 1]
            image_matches.append([sift_keys1[index1][0], sift_keys1[index1]
                                  [1], sift_keys2[index2][0], sift_keys2[index2][1]])
    return np.array(image_matches, dtype=np.float32)


def compute_fmat_error(f, image_matches, homogeneous=False):
    points1 = image_matches[:, :2]
    points2 = image_matches[:, 2:4]
    assert points1.shape == points2.shape
    if not homogeneous:
        ones = np.ones(shape=[points1.shape[0],1], dtype=points1.dtype)
        points1 = np.concatenate((points1, ones), axis=1)
        points2 = np.concatenate((points2, ones), axis=1)
    epi_lines = np.matmul(f, points1.transpose())
    dist_p2l = np.abs(np.sum(np.multiply(epi_lines.transpose(), points2), axis=1))
    dist_div = np.sqrt(np.multiply(
        epi_lines[0, :], epi_lines[0, :]) + np.multiply(epi_lines[1, :], epi_lines[1, :])) + 1e-6
    dist_p2l = np.divide(dist_p2l, dist_div.transpose())
    ave_p2l_error = np.mean(dist_p2l)
    return ave_p2l_error


if __name__ == '__main__':
    sift_list = []
    with open(sift_list_path) as f:
        lines = f.readlines()
        for line in lines:
            sift_list.append(line.strip())

    match_files = glob.glob(os.path.join(match_folder, '*.mat'))
    sift_list.sort()
    match_files.sort()

    # read all sift at once
    sift_file_map = {}
    count = 0
    for sift_file in sift_list:
        sift_name = os.path.splitext(os.path.split(sift_file)[1])[0]
        # keypoint: (x, y, color, scale, orientation)
        keypts, _ = read_feature_repo(sift_file)
        sift_file_map[sift_name] = (count, keypts)
        count = count+1

    print("Read all sift files")

    for one_mat_file in match_files:
        print("Read", one_mat_file)
        match_ret = read_match_repo(one_mat_file)
        sift_name1 = os.path.splitext(os.path.split(one_mat_file)[1])[0]
            
        for i in range(len(match_ret)):
            sift_name2 = match_ret[i][0]
            match_num = match_ret[i][1]
            hmat = match_ret[i][2]
            fmat = match_ret[i][3]
            match_pairs = match_ret[i][4]
            image_coords = get_inlier_image_coords(
                sift_file_map[sift_name1][1], sift_file_map[sift_name2][1], match_pairs, 'f')
            assert len(image_coords) == match_num[2]
            ave_error = compute_fmat_error(fmat, image_coords, homogeneous=False)
