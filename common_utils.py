"""
Tianwei Shen, HKUST, 2018
Common utility functions
"""
import os
import numpy as np
from preprocess_matches import read_feature_repo, read_match_repo, get_inlier_image_coords, compute_fmat_error

def complete_batch_size(input_list, batch_size):
    left = len(input_list) % batch_size
    if left != 0:
        for _ in range(batch_size-left):
            input_list.append(input_list[-1])
    return input_list


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False


def load_match_func(sift_folder, match_folder, frame_id, zoom_x, zoom_y, seq_length, sample_num=100, finlier_thresh=4):
    matches = []
    sift_file1 = os.path.join(sift_folder, frame_id+'.sift')
    sift_keys1, _ = read_feature_repo(sift_file1)
    half_offset = int((seq_length - 1)/2)
    frame_id_int = int(frame_id.lstrip('0'))
    for o in range(-half_offset, half_offset+1):
        adj_frame_idx = frame_id_int + o
        adj_frame_idx_str = str(adj_frame_idx)
        prefix = ['0' for i in range(len(frame_id) - len(adj_frame_idx_str))]
        adj_frame_idx_str = ''.join(prefix) + adj_frame_idx_str
        sift_file2 = os.path.join(sift_folder, adj_frame_idx_str+'.sift')
        sift_keys2, _ = read_feature_repo(sift_file2)
        if o == 0:
            continue
        elif o < 0:
            match_file = os.path.join(match_folder, adj_frame_idx_str+'.mat')
            search_sift_name = os.path.splitext(os.path.split(sift_file1)[1])[0]
        else:
            match_file = os.path.join(match_folder, frame_id+'.mat')
            search_sift_name = os.path.splitext(os.path.split(sift_file2)[1])[0]
         
        match_ret = read_match_repo(match_file)
        found = False
        for i in range(len(match_ret)):
            sift_name = match_ret[i][0]
            if sift_name == search_sift_name:
                found = True
                match_num = match_ret[i][1]
                fmat = match_ret[i][3]
                match_pairs = match_ret[i][4]
                if o < 0:
                    image_coords = get_inlier_image_coords(sift_keys2, sift_keys1, match_pairs)
                else:
                    image_coords = get_inlier_image_coords(sift_keys1, sift_keys2, match_pairs)
                ave_error = compute_fmat_error(fmat, image_coords, homogeneous=False)
                assert image_coords.shape[0] == match_num[2]
                assert ave_error < finlier_thresh
                # sample matches 
                if image_coords.shape[0] > sample_num:
                    sample_idx = np.random.choice(image_coords.shape[0], sample_num, replace=False)
                else:
                    sample_idx = range(image_coords.shape[0])
                    for i in range(sample_num - image_coords.shape[0]):
                        sample_idx.append(0)
                assert len(sample_idx) == sample_num
                sampled_coords = image_coords[sample_idx, :]
                if o < 0:
                    sampled_coords = np.matrix(
                        [zoom_x*sampled_coords[:, 2], zoom_y*sampled_coords[:, 3], zoom_x*sampled_coords[:, 0], zoom_y*sampled_coords[:, 1]]).transpose()
                else:
                    sampled_coords = np.matrix(
                        [zoom_x*sampled_coords[:, 0], zoom_y*sampled_coords[:, 1], zoom_x*sampled_coords[:, 2], zoom_y*sampled_coords[:, 3]]).transpose()
                matches.append(sampled_coords)

        if not found:
            print('Error: No matches for ', sift_file1, sift_file2)
            exit(-1)
    return matches