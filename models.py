# Copyright 2020 Tiziano Portenier
# Computer-assisted Applications in Medicine Group, Computer Vision Lab, ITET, ETH Zurich

import tensorflow as tf
import numpy as np


def _get_weight(s, layer_idx, scope, gain=np.sqrt(2), lr_mul=1., eq_lr=True):
    # he init
    fan_in = np.prod(s[:-1])  # s is [k_s, k_s, n_chan_in, n_chan_out] or [k_s, k_s, k_s, n_chan_in, n_chan_out]
    he_stddev = gain / np.sqrt(fan_in)

    if eq_lr:
        init_stddev = 1. / lr_mul
        rt_coeff = he_stddev * lr_mul
    else:
        init_stddev = he_stddev / lr_mul
        rt_coeff = lr_mul

    initializer = tf.initializers.random_normal(0, init_stddev)

    return tf.get_variable(scope + '_weights_' + str(layer_idx),
                           shape=s, initializer=initializer, dtype=tf.float32) * rt_coeff


def _bias(x, scope, layer_idx=-1, lr_mul=1., conv=True, vol=False):
    b = tf.get_variable(scope + '_biases_' + str(layer_idx),
                        shape=[x.shape[-1].value], initializer=tf.initializers.zeros()) * lr_mul
    if conv:
        return x + tf.reshape(b, [1, 1, 1, -1])
    else:
        return x + b


def downsample(t_in, factor=2):
    """
    Average pool downsampling
    """
    t_out = tf.nn.avg_pool(t_in, [1, factor, factor, 1], [1, factor, factor, 1], padding='VALID',
                           data_format='NHWC')
    return t_out


def conv2d(t_in, n_chan, layer_idx=-1, scope='conv', kernel_size=3, stride=1, gain=np.sqrt(2), lr_mul=1.,
           eq_lr=True):
    """
    A 2D convolutional layer

    """
    w = _get_weight([kernel_size, kernel_size, t_in.shape[-1].value, n_chan], layer_idx, scope, gain=gain,
                    lr_mul=lr_mul, eq_lr=eq_lr)

    return tf.nn.conv2d(t_in, w, strides=[stride, stride], padding='SAME', data_format="NHWC")


def fc(t_in, n_out, layer_idx=-1, scope='fc', gain=np.sqrt(2), lr_mul=1., eq_lr=True):
    """
    A dense (fully connected) layer
    """
    w = _get_weight([t_in.shape[-1].value, n_out], layer_idx, scope, gain=gain, lr_mul=lr_mul,
                    eq_lr=eq_lr)

    return tf.matmul(t_in, w)


def sample_trilinear(t, coords, img_h=128, img_w=128):
    """
    Samples noise octaves in one shot

    :param t: noise cube [n_octaves, noise_res, noise_res, noise_res] (same noise foreach sample in batch)
    :param coords: octave-transformed sampling positions [bs, n_octaves, 3, img_h*img_w]
    :param img_h: height of image to synthesize
    :param img_w: width of image to synthesize
    :return: sampled noise octaves [bs, n_octaves, img_h, img_w]
    """
    # in- and output dimensions
    n_octaves, noise_res = t.get_shape().as_list()[:2]
    bs = coords.get_shape().as_list()[0]

    # all contributing source coordinates (interpolation endpoints)
    x0 = tf.floor(coords[:, :, 0, :])
    x1 = x0 + 1
    y0 = tf.floor(coords[:, :, 1, :])
    y1 = y0 + 1
    z0 = tf.floor(coords[:, :, 2, :])
    z1 = z0 + 1

    # interpolation weights
    w_x = coords[:, :, 0, :] - x0
    w_y = coords[:, :, 1, :] - y0
    w_z = coords[:, :, 2, :] - z0

    # modulo for out-of-bound indices
    x0 = tf.floormod(x0, tf.ones_like(x0) * noise_res)
    x1 = tf.floormod(x1, tf.ones_like(x1) * noise_res)
    y0 = tf.floormod(y0, tf.ones_like(y0) * noise_res)
    y1 = tf.floormod(y1, tf.ones_like(y1) * noise_res)
    z0 = tf.floormod(z0, tf.ones_like(z0) * noise_res)
    z1 = tf.floormod(z1, tf.ones_like(z1) * noise_res)

    # for mem efficiency we flatten voxels s.t. we need only one index per element instead of 3
    t = tf.reshape(t, [n_octaves, noise_res**3])

    # index arrays (in flattened voxel array)
    idx_x0_y0_z0 = tf.cast(y0 * noise_res + x0 * noise_res**2 + z0, tf.int32)
    idx_x0_y0_z1 = tf.cast(y0 * noise_res + x0 * noise_res**2 + z1, tf.int32)
    idx_x0_y1_z0 = tf.cast(y1 * noise_res + x0 * noise_res**2 + z0, tf.int32)
    idx_x0_y1_z1 = tf.cast(y1 * noise_res + x0 * noise_res**2 + z1, tf.int32)
    idx_x1_y0_z0 = tf.cast(y0 * noise_res + x1 * noise_res**2 + z0, tf.int32)
    idx_x1_y0_z1 = tf.cast(y0 * noise_res + x1 * noise_res**2 + z1, tf.int32)
    idx_x1_y1_z0 = tf.cast(y1 * noise_res + x1 * noise_res**2 + z0, tf.int32)
    idx_x1_y1_z1 = tf.cast(y1 * noise_res + x1 * noise_res**2 + z1, tf.int32)

    def _gather(idx):
        # TODO: not quite efficient. ;)
        out = []
        for i in range(n_octaves):
            g = tf.gather(t[i], idx[:, i, :], axis=0, batch_dims=1)
            out.append(tf.expand_dims(g, 1))

        return tf.concat(out, 1)

    # gather contributing samples --> now 2D!
    val_x0_y0_z0 = _gather(idx_x0_y0_z0)
    val_x0_y0_z1 = _gather(idx_x0_y0_z1)
    val_x0_y1_z0 = _gather(idx_x0_y1_z0)
    val_x0_y1_z1 = _gather(idx_x0_y1_z1)
    val_x1_y0_z0 = _gather(idx_x1_y0_z0)
    val_x1_y0_z1 = _gather(idx_x1_y0_z1)
    val_x1_y1_z0 = _gather(idx_x1_y1_z0)
    val_x1_y1_z1 = _gather(idx_x1_y1_z1)

    # interpolate along z ...
    c_00 = val_x0_y0_z0 * (1.0 - w_z) + val_x0_y0_z1 * w_z
    c_01 = val_x0_y1_z0 * (1.0 - w_z) + val_x0_y1_z1 * w_z
    c_10 = val_x1_y0_z0 * (1.0 - w_z) + val_x1_y0_z1 * w_z
    c_11 = val_x1_y1_z0 * (1.0 - w_z) + val_x1_y1_z1 * w_z

    # ... along y ...
    c_0 = c_00 * (1.0 - w_y) + c_01 * w_y
    c_1 = c_10 * (1.0 - w_y) + c_11 * w_y

    # ... and along x
    c = c_0 * (1.0 - w_x) + c_1 * w_x

    # reshape
    c = tf.reshape(c, [bs, n_octaves, img_h, img_w])

    return c


def sampler_single(noise_cube, coords, transformations, img_size=128, img_h=None, img_w=None,
                   act=tf.nn.leaky_relu, scope='sampler', hidden_dim=128, eq_lr=True):
    """
    MLP that maps 3D coordinate to an rgb texture value
    :param noise_cube: noise, same foreach sample in batch [n_octaves, noise_res, noise_res, noise_res]
    :param coords: xyz coords of slices to synthesize [bs, 3, img_size^2)
    :param transformations: octave transformations [bs, n_octaves, 3, 3]
    :param img_size: training patch size
    :param img_h: height of output image (for inference)
    :param img_w: width of output image (for inference)
    :param act: activation function
    :param scope: scope name
    :param hidden_dim: number of neurons in dense layers
    :param eq_lr: whether to use equalized learning rate
    :return: rgb image
    """

    def _add_noise(t_in, noise, layer_idx):
        """
        Add noise in 4 layers. In each layer, add 4 (n_octaves=16) or 8 (n_octaves=32) octaves
        :param t_in: pre-activated layer output
        :param noise: noise octaves
        :param layer_idx: layer index
        :return: pre-activated layer output with added noise octaves
        """

        t_out = t_in
        if n_octaves == 16:
            per_octave_w = tf.get_variable('octave_weights_' + str(layer_idx), trainable=True,
                                           shape=[4, hidden_dim], initializer=tf.random_normal_initializer())
            for i in range(4):
                t_out = t_out + noise[:, :, :, layer_idx * 4 + i:layer_idx * 4 + i + 1] * \
                        tf.reshape(per_octave_w[i, :], [1, 1, 1, hidden_dim])
        elif n_octaves == 32:
            per_octave_w = tf.get_variable('octave_weights_' + str(layer_idx), trainable=True,
                                           shape=[8, hidden_dim], initializer=tf.random_normal_initializer())
            for i in range(8):
                t_out = t_out + noise[:, :, :, layer_idx * 8 + i:layer_idx * 8 + i + 1] * \
                        tf.reshape(per_octave_w[i, :], [1, 1, 1, hidden_dim])
        else:
            import sys
            sys.exit("oops, need n_octaves either 16 or 32!")

        return t_out

    def post_conv(t_in, noise, layer_idx=-1):
        """
        Adds octave noise and bias, applies activation function
        """
        t_out = _add_noise(t_in, noise, layer_idx)
        t_out = _bias(t_out, 'conv', layer_idx)
        t_out = act(t_out)

        return t_out

    if img_h is None:
        img_h = img_size
    if img_w is None:
        img_w = img_size

    n_octaves = noise_cube.get_shape().as_list()[0]

    # initialize octaves to 2**i for n_octaves = 16 and 2**(i/2) otherwise
    oct_factors = np.zeros((n_octaves, 3, 3), dtype=np.float32)
    for i in range(n_octaves):
        for j in range(3):
            if n_octaves == 16:
                oct_factors[i, j, j] = 2 ** i
            else:
                oct_factors[i, j, j] = 2 ** (i/2.)

    oct_factors = tf.convert_to_tensor(oct_factors)
    transformations = tf.matmul(transformations, oct_factors)

    # obtain octave sampling coordinates by transforming slice coordinates with octave matrices
    # broadcast (same input coords foreach octave)
    coords = tf.expand_dims(coords, 1)
    coords = tf.tile(coords, [1, n_octaves, 1, 1])
    # now we have a distinct set of coordinates foreach batch and octave
    octave_coords = tf.matmul(transformations, coords)

    # sample noise
    sampled_noise = sample_trilinear(noise_cube, octave_coords, img_h=img_h, img_w=img_w)
    # noise in last dimension (NHWC)
    sampled_noise = tf.transpose(sampled_noise, [0, 2, 3, 1])

    with tf.variable_scope(scope):
        # start with a learned constant
        base = tf.get_variable('constant_input', shape=[1, 1, 1, hidden_dim], trainable=True,
                               initializer=tf.initializers.ones())
        # broadcast (constant per pixel and per sample)
        bs = coords.shape[0].value
        base = tf.tile(base, [bs, img_h, img_w, 1])

        layer_idx = 0
        l0 = conv2d(base, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l0 = post_conv(l0, sampled_noise, layer_idx=layer_idx)

        layer_idx = 1
        l1 = conv2d(l0, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l1 = post_conv(l1, sampled_noise, layer_idx=layer_idx)

        layer_idx = 2
        l2 = conv2d(l1, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l2 = post_conv(l2, sampled_noise, layer_idx=layer_idx)

        layer_idx = 3
        l3 = conv2d(l2, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l3 = post_conv(l3, sampled_noise, layer_idx=layer_idx)

        # no noise in final hidden layer
        layer_idx = 4
        l4 = conv2d(l3, hidden_dim, layer_idx=layer_idx, kernel_size=1,  eq_lr=eq_lr)
        l4 = _bias(l4, 'conv', layer_idx)
        l4 = act(l4)

        # 2rgb linear
        layer_idx = 5
        rgb = conv2d(l4, 3, layer_idx=layer_idx, kernel_size=1, gain=1., eq_lr=eq_lr)
        rgb = _bias(rgb, 'conv', layer_idx)

        return rgb


def discriminator(img, reuse=True, act=tf.nn.leaky_relu, scope='discriminator', eq_lr=True):

    def post_conv(t_in, layer_idx=-1):
        """
        Applies bias and activation function
        """
        t_out = _bias(t_in, 'conv', layer_idx)
        t_out = act(t_out)
        return t_out

    with tf.variable_scope(scope, reuse=reuse):
        # conv layer 0 [bs, 128, 128, 32]
        layer_idx = 0
        l0 = conv2d(img, 32, layer_idx=layer_idx, eq_lr=eq_lr)
        l0 = post_conv(l0, layer_idx=layer_idx)

        # conv layer 1 [bs, 64, 64, 64]
        layer_idx = 1
        l1 = downsample(l0, factor=2)
        l1 = conv2d(l1, 64, layer_idx=layer_idx, eq_lr=eq_lr)
        l1 = post_conv(l1, layer_idx=layer_idx)

        # conv layer 2 [bs, 32, 32, 128]
        layer_idx = 2
        l2 = downsample(l1, factor=2)
        l2 = conv2d(l2, 128, layer_idx=layer_idx, eq_lr=eq_lr)
        l2 = post_conv(l2, layer_idx=layer_idx)

        # conv layer 3 [bs, 16, 16, 256]
        layer_idx = 3
        l3 = downsample(l2, factor=2)
        l3 = conv2d(l3, 256, layer_idx=layer_idx, eq_lr=eq_lr)
        l3 = post_conv(l3, layer_idx=layer_idx)

        # conv layer 4 [bs, 8, 8, 256]
        layer_idx = 4
        l4 = downsample(l3, factor=2)
        l4 = conv2d(l4, 256, layer_idx=layer_idx, eq_lr=eq_lr)
        l4 = post_conv(l4, layer_idx=layer_idx)

        # conv layer 5 [bs, 4, 4, 512]
        layer_idx = 5
        l5 = downsample(l4, factor=2)
        l5 = conv2d(l5, 512, layer_idx=layer_idx, eq_lr=eq_lr)
        l5 = post_conv(l5, layer_idx=layer_idx)

        # conv layer 6 [bs, 2, 2, 512]
        layer_idx = 6
        l6 = downsample(l5, factor=2)
        l6 = conv2d(l6, 512, layer_idx=layer_idx, eq_lr=eq_lr)
        l6 = post_conv(l6, layer_idx=layer_idx)

        # dense layer 7 [bs, 512]
        layer_idx = 7
        l7 = downsample(l6, factor=2)
        l7 = tf.reshape(l7, [-1, 512])
        l7 = fc(l7, 512, layer_idx=layer_idx, eq_lr=eq_lr)
        l7 = _bias(l7, 'fc', layer_idx=layer_idx, conv=False)
        l7 = act(l7)

        # output layer (dense linear)
        layer_idx = 8
        logits = fc(l7, 1, layer_idx=layer_idx, gain=1., eq_lr=eq_lr)
        logits = _bias(logits, 'fc', layer_idx=layer_idx, conv=False)

    return logits, [l0, l1, l2, l3, l4, l5, l6]
