"""
Tianwei Shen, HKUST, 2018 - 2019.
DeepSlam class defines the training procedure and losses
"""
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from geo_utils import get_relative_pose, projective_inverse_warp, pose_vec2mat, mat2euler, \
    fundamental_matrix_from_rt, reprojection_error

class DeepSlam(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        '''[summary]
        build training graph
        
        Returns:
            data loader and batch sample for train() to initialize
            undefined placeholders
        '''

        opt = self.opt
        is_read_pose = opt.with_pose or opt.pose_weight > 0
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales,
                            is_read_pose,
                            opt.match_num)
        with tf.name_scope("data_loading"):
            batch_sample = loader.load_train_batch()
            # give additional batch_size info since the input is undetermined placeholder
            inputs = batch_sample.get_next()
            tgt_image = inputs[0]
            src_image_stack = inputs[1]
            intrinsics = inputs[2]
            #[bs, 128, 416, 3]
            tgt_image.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
            # [bs, 128, 416, 6]
            src_image_stack.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3*opt.num_source])
            # [bs, 4, 3, 3]
            intrinsics.set_shape([opt.batch_size, opt.num_scales, 3, 3])
            if is_read_pose:
                poses = inputs[3]
                poses.set_shape([opt.batch_size, 3, 6])
            if opt.match_num > 0:
                matches = inputs[3]
                matches.set_shape([opt.batch_size, opt.num_source, opt.match_num, 4])
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

        with tf.name_scope("depth_prediction"):
            pred_disp, _ = disp_net_res50(tgt_image, is_training=True)
            if opt.with_pose:   # cannot normalize pose here due to given scale
                pred_depth = [1. / d for d in pred_disp]
            else:
                pred_depth = [1. / self.spatial_normalize(d) for d in pred_disp]

        with tf.name_scope("pose_and_explainability_prediction"):
            pred_poses, _ = pose_net(tgt_image, src_image_stack, is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            smooth_loss = 0
            pose_loss = 0
            ssim_loss = 0
            match_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            mask_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            for s in range(opt.num_scales):
                # Scale the source and target images for computing loss at the according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])                
                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp[s], curr_tgt_image)

                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    if is_read_pose:
                        relative_pose = get_relative_pose(poses[:,0,:], poses[:,i+1,:])
                        relative_rot = tf.slice(relative_pose, [0, 0, 0], [-1, 3, 3])
                        relative_rot_vec = mat2euler(relative_rot)
                        relative_trans_vec = tf.slice(relative_pose, [0, 0, 3], [-1, 3, 1])
                        relative_pose_vec = tf.squeeze(tf.concat([relative_rot_vec, relative_trans_vec], axis=1))
                    
                    if opt.with_pose:
                        warp_pose = relative_pose
                        pose_is_vec = False
                    else:
                        warp_pose = pred_poses[:,i,:]
                        pose_is_vec = True
                    
                    curr_proj_image, mask = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        tf.squeeze(pred_depth[s], axis=3), 
                        warp_pose, intrinsics[:,s,:,:], is_vec=pose_is_vec)
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    curr_proj_error = tf.multiply(curr_proj_error, mask)

                    # below-threshold mask
                    perct_thresh = tf.contrib.distributions.percentile(curr_proj_error, q=99, axis=[1,2])
                    perct_thresh = tf.expand_dims(tf.expand_dims(perct_thresh, 1), 1)
                    curr_proj_error = tf.clip_by_value(curr_proj_error, 0, perct_thresh)
                    above_perct_thresh_region = tf.reduce_max(tf.cast(tf.equal(curr_proj_error, perct_thresh), 'float32'), axis=3)
                    above_perct_thresh_region = tf.greater_equal(above_perct_thresh_region, 1.0)
                    suppresion_mask = tf.expand_dims(1.0 - tf.cast(above_perct_thresh_region, 'float32'), axis=3)
                    curr_proj_error = tf.multiply(curr_proj_error, suppresion_mask)
                    mask = tf.multiply(mask, suppresion_mask)
                    pixel_loss += tf.reduce_mean(curr_proj_error) 

                    # SSIM loss
                    if opt.ssim_weight > 0:
                        ssim_mask = slim.avg_pool2d(mask, 3, 1, 'VALID')
                        ssim_loss += tf.reduce_mean(
                            ssim_mask * self.compute_ssim_loss(curr_proj_image, curr_tgt_image))

                    # Relative pose error
                    if opt.pose_weight > 0 and s == 0:  # only do it for highest resolution
                        pose_loss += tf.reduce_mean(self.compute_pose_loss(relative_pose_vec, pred_poses[:, i, :]))
                    
                    # Matches loss (fundamental matrix)
                    if opt.match_num > 0 and s == 0:  # only do it for highest resolution
                        match_loss += self.compute_match_loss(matches[:, i, :, :], tf.squeeze(
                            pred_depth[s], axis=3), pred_poses[:, i, :], intrinsics[:, s, :, :])

                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        mask_stack = mask
                        proj_error_stack = curr_proj_error
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, curr_proj_image], axis=3)
                        mask_stack = tf.concat([mask_stack, mask], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, curr_proj_error], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                mask_stack_all.append(mask_stack)
                proj_error_stack_all.append(proj_error_stack)
            total_loss = opt.ssim_weight * ssim_loss + \
                (1 - opt.ssim_weight) * pixel_loss + \
                smooth_loss + opt.pose_weight * pose_loss + opt.match_weight * match_loss

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.pose_loss = pose_loss
        self.smooth_loss = smooth_loss
        self.ssim_loss = ssim_loss
        self.match_loss = match_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.mask_stack_all = mask_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        return loader, batch_sample


    def compute_smooth_loss(self, disp, img):
        def _gradient(pred):
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            return D_dx, D_dy

        disp_gradients_x, disp_gradients_y = _gradient(disp)
        image_gradients_x, image_gradients_y = _gradient(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))
    

    def compute_pose_loss(self, prior_pose_vec, pred_pose_vec):
        rot_vec_err = tf.norm(prior_pose_vec[:,:3] - pred_pose_vec[:,:3], axis=1)
        trans_err = tf.norm(tf.nn.l2_normalize(
            prior_pose_vec[:, 3:], dim=1) - tf.nn.l2_normalize(pred_pose_vec[:, 3:], dim=1), axis=1)
        return rot_vec_err + trans_err


    # reference https://github.com/tensorflow/models/tree/master/research/vid2depth/model.py
    def compute_ssim_loss(self, x, y):
        """Computes a differentiable structured image similarity measure."""
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
        sigma_x = slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
        sigma_y = slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        return tf.clip_by_value((1 - ssim) / 2, 0, 1)

    
    # reference: https://github.com/yzcjtr/GeoNet/blob/master/geonet_model.py
    # and https://arxiv.org/abs/1712.00175
    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean
    

    def normalize_for_show(self, disp, thresh=90):
        disp_max = tf.contrib.distributions.percentile(disp, q=thresh, axis=[1,2])
        disp_max = tf.expand_dims(tf.expand_dims(disp_max, 1), 1)
        clip_disp = tf.clip_by_value(disp, 0, disp_max)
        return clip_disp


    def compute_match_loss(self, matches, pred_depth, pose, intrinsics):
        points1 = tf.slice(matches, [0, 0, 0], [-1, -1, 2])
        points2 = tf.slice(matches, [0, 0, 2], [-1, -1, 2])
        ones = tf.ones([self.opt.batch_size, self.opt.match_num, 1])
        points1 = tf.concat([points1, ones], axis=2)
        points2 = tf.concat([points2, ones], axis=2)
        match_num = matches.get_shape().as_list()[1]

        # compute fundamental matrix loss
        fmat = fundamental_matrix_from_rt(pose, intrinsics)
        fmat = tf.expand_dims(fmat, axis=1)
        fmat_tiles = tf.tile(fmat, [1, match_num, 1, 1])
        epi_lines = tf.matmul(fmat_tiles, tf.expand_dims(points1, axis=3))
        dist_p2l = tf.abs(tf.matmul(tf.transpose(epi_lines, perm=[0, 1, 3, 2]), tf.expand_dims(points2, axis=3)))

        a = tf.slice(epi_lines, [0,0,0,0], [-1,-1,1,-1])
        b = tf.slice(epi_lines, [0,0,1,0], [-1,-1,1,-1])
        dist_div = tf.sqrt(a*a + b*b) + 1e-6
        dist_p2l = tf.reduce_mean(dist_p2l / dist_div)
        return dist_p2l


    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        if opt.smooth_weight > 0:
            tf.summary.scalar("smooth_loss", self.smooth_loss)
        if opt.pose_weight > 0:
            tf.summary.scalar("pose_loss", self.pose_loss)
        if opt.ssim_weight > 0:
            tf.summary.scalar("ssim_loss", self.ssim_loss)
        if opt.match_num > 0:
            tf.summary.scalar("match_loss", self.match_loss)
        #for s in range(opt.num_scales):
        s = 0   # only show the error images of the highest resolution (scale 0)
        tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
        shown_disparity_image = self.normalize_for_show(1./self.pred_depth[s])
        tf.summary.image('scale%d_disparity_image' % s, shown_disparity_image)
        tf.summary.image('scale%d_target_image' % s, self.deprocess_image(self.tgt_image_all[s]))
        for i in range(opt.num_source):
            tf.summary.image(
                'scale%d_source_image_%d' % (s, i),
                self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
            proj_images = self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3])
            mask_images = self.mask_stack_all[s][:, :, :, i:i+1]
            proj_error_images = self.deprocess_image(tf.clip_by_value(
                self.proj_error_stack_all[s][:, :, :, i*3:(i+1)*3] - 1, -1, 1))
            tf.summary.image('scale%d_projected_image_%d' % (s, i), proj_images)
            tf.summary.image('scale%d_proj_error_%d' % (s, i), proj_error_images)
            tf.summary.image('scale%d_mask_%d' % (s, i), mask_images)


    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        self.opt = opt
        if opt.match_num > 0:  # don't use match and pose at the same time
            opt.with_pose = False
        data_loader, batch_sample = self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=None)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            data_loader.init_data_pipeline(sess, batch_sample)
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_ckpt_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_ckpt_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(0, opt.max_steps):
                fetches = {"train": self.train_op,
                           "global_step": self.global_step,
                           "incr_global_step": self.incr_global_step}

                if step % opt.summary_freq == 0:
                    fetches["total_loss"] = self.total_loss
                    fetches["pixel_loss"] = self.pixel_loss
                    fetches["smooth_loss"] = self.smooth_loss
                    fetches["summary"] = sv.summary_op
                    if opt.pose_weight > 0:
                        fetches["pose_loss"] = self.pose_loss

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f"
                          % (train_epoch, train_step, self.steps_per_epoch,
                             (time.time() - start_time)/opt.summary_freq))
                    print("total/pixel/smooth loss: [%.3f/%.3f/%.3f]\n" % (
                        results["total_loss"], results["pixel_loss"], results["smooth_loss"]))
                    start_time = time.time()

                # save model
                if step != 0 and step % opt.save_freq == 0:
                    self.save(sess, opt.checkpoint_dir, gs-1)


    def select_tensor_or_placeholder_input(self, input_uint8):
        if input_uint8 == None:
            input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                        self.img_height, self.img_width, 3], name='raw_input')
            self.inputs = input_uint8
        else:
            self.inputs = None
        input_mc = self.preprocess_image(input_uint8)
        return input_mc


    def build_depth_test_graph(self, input_uint8):
        input_mc = self.select_tensor_or_placeholder_input(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net_res50(input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints


    def build_pose_test_graph(self, input_uint8):
        input_mc = self.select_tensor_or_placeholder_input(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _ = pose_net(tgt_image, src_image_stack, is_training=False)
            self.pred_poses = pred_poses


    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.


    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)


    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1,
                        input_img_uint8=None):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph(input_img_uint8)
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph(input_img_uint8)


    def inference(self, sess, mode, inputs=None):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        if inputs is None:
            results = sess.run(fetches)
        else:
            results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results


    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint step %d to %s..." % (step, checkpoint_dir))
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
