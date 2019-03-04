"""
Tianwei Shen, HKUST, 2018 - 2019
Geometric utility functions
Adapted from: https://github.com/tinghuiz/SfMLearner
"""

from __future__ import division
import numpy as np
import math
import tensorflow as tf

def euler2mat(z, y, x):
    """[summary] Converts euler angles to rotation matrix
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
    
    Arguments:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """

    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


def mat2euler(rot):
    r00 = tf.slice(rot, [0, 0, 0], [-1, 1, 1])
    r01 = tf.slice(rot, [0, 0, 1], [-1, 1, 1])
    r02 = tf.slice(rot, [0, 0, 2], [-1, 1, 1])
    r10 = tf.slice(rot, [0, 1, 0], [-1, 1, 1])
    r11 = tf.slice(rot, [0, 1, 1], [-1, 1, 1])
    r12 = tf.slice(rot, [0, 1, 2], [-1, 1, 1])
    r22 = tf.slice(rot, [0, 2, 2], [-1, 1, 1])
    cy = tf.sqrt(r22*r22 + r12 * r12)

    def f1():
        z = tf.atan2(-r01, r00)    
        y = tf.atan2(r02, cy)
        x = tf.atan2(-r12, r22)
        return tf.concat([z,y,x], axis=1)

    def f2():
        z = tf.atan2(r10, r11)
        y = tf.atan2(r02, cy)
        x = tf.zeros_like(y)
        return tf.concat([z,y,x], axis=1)
    
    x1 = f1()
    x2 = f2()
    return tf.where(tf.squeeze(tf.less(cy, 1e-6), axis=[1,2]), x2, x1)


def pose_vec2rt(vec):
    """Converts 6DoF parameters to rotation matrix (bs,3,3) and translation vector (bs,3,1)"""
    translation = tf.slice(vec, [0, 3], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rz = tf.slice(vec, [0, 0], [-1, 1])
    ry = tf.slice(vec, [0, 1], [-1, 1])
    rx = tf.slice(vec, [0, 2], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    return rot_mat, translation


def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of [rz, ry, rx, tx, ty, tz] -- [B, 6]
        (NOT the original SfMLearner: tx, ty, tz, rx, ry, rz -- [B, 6])
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.get_shape().as_list()
    rot_mat, translation = pose_vec2rt(vec)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
      intrinsics: camera intrinsics [batch, 3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height*width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
      cam_coords: [batch, 4, height, width]
      proj: [batch, 4, 4]
    Returns:
      Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def get_relative_pose(target_pose_vec, source_pose_vec):
    """[summary]
    get relative pose from two absolution poses
    Arguments:
        target_pose_vec {[batch_size, 6]} -- [rz, ry, rx, tx, ty, tz]
        source_pose_vec {[batch_size, 6]} -- [rz, ry, rx, tx, ty, tz]
        both are abosulte (up to scale) pose w.r.t. the world coord
        Note: this is different from the original SfMLearner which is [tx, ty, tz, rx, ry, rz].
    Returns:
        transformation from target to source
    """
    target_pose = pose_vec2mat(target_pose_vec)
    source_pose = pose_vec2mat(source_pose_vec)

    target_pose_inv = tf.linalg.inv(target_pose)
    transform_mat = tf.matmul(source_pose, target_pose_inv)
    return transform_mat


def projective_inverse_warp(img, depth, pose, intrinsics, is_vec=True):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 6], in the
            order of tx, ty, tz, rx, ry, rz
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Convert pose vector to matrix
    if is_vec:
        pose = pose_vec2mat(pose)

    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img, mask = bilinear_sampler(img, src_pixel_coords)
    return output_img, mask


def reprojection_error(src_points, tgt_points, depth, pose, intrinsics):
    """Compute re-projection error given match points
    
    Arguments:
        src_points {[type]} -- homogeneous coordinates [batch_size, match_num, 3]
        tgt_points {[type]} -- homogeneous coordinates [batch_size, match_num, 3]
        depth {depth mat} -- [batch_size, height, width]
        pose {pose vector} -- [batch_size, 6]
        intrinsics {[type]} -- [batch_size, 3, 3]
    
    Returns:
        reprojection error given sparse matches
    """

    pose = pose_vec2mat(pose)
    #cam_coords = pixel2cam(depth, intrinsics)

    exit(-1)
    return 0


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(_repeat(tf.cast(tf.range(
            coord_size[0]), 'float32') * dim1, coord_size[1] * coord_size[2]), [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([w00 * im00, w01 * im01, w10 * im10, w11 * im11])
        mask = tf.add_n([w00, w01, w10, w11])
        return output, mask


def skew_symmetric_mat3(vec3):
    """compute the skew symmetric matrix for cross product
    
    Arguments:
        vec {vector of shape [batch_size, 3, 1]}
    """
    bs = tf.shape(vec3)[0]      # batch size
    a1 = tf.slice(vec3, [0,0,0], [-1,1,-1])
    a2 = tf.slice(vec3, [0,1,0], [-1,1,-1])
    a3 = tf.slice(vec3, [0,2,0], [-1,1,-1])
    zeros = tf.zeros([bs, 1, 1])
    row1 = tf.concat([zeros, -a3, a2], axis=2)
    row2 = tf.concat([a3, zeros, -a1], axis=2)
    row3 = tf.concat([-a2, a1, zeros], axis=2)
    vec3_ssm = tf.concat([row1, row2, row3], axis=1)
    return vec3_ssm


def fundamental_matrix_from_rt(vec, intrinsics):
    rot_mat, translation = pose_vec2rt(vec)
    translation_ssm = skew_symmetric_mat3(translation)
    essential_mat = tf.matmul(rot_mat, translation_ssm, name='essential_mat')
    intrinsics_inv = tf.matrix_inverse(intrinsics)
    fundamental_mat = tf.matmul(intrinsics_inv, essential_mat, transpose_a=True)
    fundamental_mat = tf.matmul(fundamental_mat, intrinsics_inv)
    return fundamental_mat

#============================== numpy functions ==============================
def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0,0] *= sx
    out[0,2] *= sx
    out[1,1] *= sy
    out[1,2] *= sy
    return out