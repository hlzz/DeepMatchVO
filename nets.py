from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_RESNET50_SCALING = 5      # should set to 50 to use the gt pose
MIN_DISP = 0.01

def get_pred(x, scale, offset):
    disp = scale * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + offset
    return disp


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def pose_net(tgt_image, src_image_stack, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, end_points


# Adapt from https://github.com/yzcjtr/GeoNet/blob/master/geonet_nets.py
def disp_net_res50(tgt_image, is_training=True):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(10e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            conv1 = conv(tgt_image, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            
            # DECODING
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_pred(iconv4, DISP_RESNET50_SCALING, MIN_DISP)
            upred4  = upsample_nn(pred4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_pred(iconv3, DISP_RESNET50_SCALING, MIN_DISP)
            upred3  = upsample_nn(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_pred(iconv2, DISP_RESNET50_SCALING, MIN_DISP)
            upred2  = upsample_nn(pred2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_pred(iconv1, DISP_RESNET50_SCALING, MIN_DISP)

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [pred1, pred2, pred3, pred4], end_points


def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)


def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)


def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv


def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.relu(conv3 + shortcut)


def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out