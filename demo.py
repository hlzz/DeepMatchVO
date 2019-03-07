from __future__ import division
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import matplotlib.pyplot as plt
from deep_slam import DeepSlam

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    return depth

if __name__ == '__main__':
    img_height=128
    img_width=416
    ckpt_file = 'ckpt/model-250000'
    fh = open('data/example.png', 'r')
    I = pil.open(fh)
    I = I.resize((img_width, img_height), pil.ANTIALIAS)
    I = np.array(I)
    
    system = DeepSlam()
    system.setup_inference(img_height, img_width, mode='depth')
    
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        pred = system.inference(sess, mode='depth', inputs=I[None,:,:,:])
    
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow(I)
    plt.subplot(1,2,2)
    plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    plt.show()
    