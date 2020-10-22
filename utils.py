# Copyright 2020 Tiziano Portenier
# Computer-assisted Applications in Medicine Group, Computer Vision Lab, ITET, ETH Zurich

import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R


def meshgrid2D(h, w):
    x_coords = tf.linspace(0.0, w - 1, w)
    x_coords = tf.reshape(x_coords, [1, -1])
    x_coords = tf.tile(x_coords, [h, 1])
    x_coords = tf.reshape(x_coords, [1, -1])

    y_coords = tf.linspace(0.0, h - 1, h)
    y_coords = tf.reshape(y_coords, [-1, 1])
    y_coords = tf.tile(y_coords, [1, w])
    y_coords = tf.reshape(y_coords, [1, -1])

    # normalize to [-1, 1] for training patch size of 128^2
    x_coords /= 128
    x_coords -= 0.5
    x_coords *= 2.
    y_coords /= 128
    y_coords -= 0.5
    y_coords *= 2.
    # slice at z=0
    z_coords = tf.zeros_like(y_coords)

    # homogeneous coordinate (for translations)
    w_coords = tf.ones_like(y_coords)

    coords = tf.concat([x_coords, y_coords, z_coords, w_coords], 0)

    return coords


def get_random_slicing_matrices(bs, wood=False):
    """
        Samples a random transformation matrix, slice is at z=0.
        Rotate using random angles in [0, 90] and
        translate using random offsets in [-1, 1]
    """
    matrices = np.zeros((bs, 4, 4))
    for i in range(bs):
        if wood:
            # only rotate around x
            angles = np.random.uniform(low=0., high=90.0, size=1)
            r = R.from_euler('x', angles, degrees=True)
        else:
            # rotate randomly around each axis
            angles = np.random.uniform(low=0., high=90.0, size=3)
            r = R.from_euler('xyz', angles, degrees=True)

        rot = np.identity(4, dtype=np.float32)
        rot[:3, :3] = r.as_dcm()

        # translate randomly in all 3 directions
        offset = np.random.uniform(low=-1., high=1., size=3)
        trans = np.identity(4, dtype=np.float32)
        trans[0:3, 3] = offset

        # final matrix
        m = np.matmul(trans, rot)

        matrices[i, :, :] = m

    return matrices
