from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from deep_slam import DeepSlam
from data_loader import DataLoader
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
from common_utils import complete_batch_size, is_valid_sample

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Raw odometry dataset directory")
flags.DEFINE_string("concat_img_dir", None, "Preprocess image dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def load_kitti_image_sequence_names(dataset_dir, frames, seq_length):
    image_sequence_names = []
    target_inds = []
    frame_num = len(frames)
    for tgt_idx in range(frame_num):
        if not is_valid_sample(frames, tgt_idx, FLAGS.seq_length):
            continue
        curr_drive, curr_frame_id = frames[tgt_idx].split(' ')
        img_filename = os.path.join(dataset_dir, '%s/%s.jpg' % (curr_drive, curr_frame_id))
        image_sequence_names.append(img_filename)
        target_inds.append(tgt_idx)
    return image_sequence_names, target_inds


def main():
    # get input images
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    concat_img_dir = os.path.join(FLAGS.concat_img_dir, '%.2d' % FLAGS.test_seq)
    max_src_offset = int((FLAGS.seq_length - 1)/2)
    N = len(glob(concat_img_dir + '/*.jpg')) + 2*max_src_offset
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]

    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    with tf.Session() as sess:
        # setup input tensor
        loader = DataLoader(FLAGS.concat_img_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.seq_length-1)
        image_sequence_names, tgt_inds = load_kitti_image_sequence_names(FLAGS.concat_img_dir, test_frames, FLAGS.seq_length)
        image_sequence_names = complete_batch_size(image_sequence_names, FLAGS.batch_size)
        tgt_inds = complete_batch_size(tgt_inds, FLAGS.batch_size)
        assert len(tgt_inds) == len(image_sequence_names)
        batch_sample = loader.load_test_batch(image_sequence_names)
        sess.run(batch_sample.initializer)
        input_batch = batch_sample.get_next()
        input_batch.set_shape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3])

        # init system
        system = DeepSlam()
        system.setup_inference(FLAGS.img_height, FLAGS.img_width,
                               'pose', FLAGS.seq_length, FLAGS.batch_size, input_batch)
        saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
        saver.restore(sess, FLAGS.ckpt_file)

        round_num = len(image_sequence_names) // FLAGS.batch_size
        for i in range(round_num):
            pred = system.inference(sess, mode='pose')
            for j in range(FLAGS.batch_size):
                tgt_idx = tgt_inds[i * FLAGS.batch_size + j]
                pred_poses = pred['pose'][j]
                # Insert the target pose [0, 0, 0, 0, 0, 0] to the middle
                pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
                curr_times = times[tgt_idx-max_src_offset : tgt_idx+max_src_offset+1]
                out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
                dump_pose_seq_TUM(out_file, pred_poses, curr_times)

if __name__ == '__main__':
    main()
