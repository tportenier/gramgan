# Copyright 2020 Tiziano Portenier
# Computer-assisted Applications in Medicine Group, Computer Vision Lab, ITET, ETH Zurich

import tensorflow as tf
import cv2
import numpy as np

from utils import meshgrid2D, get_random_slicing_matrices
import models


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_samples", 1, "how many samples to generate")
flags.DEFINE_integer("img_w", 512, "output image patch size")
flags.DEFINE_integer("img_h",512, "output image patch size")
flags.DEFINE_integer("hidden_dim", 128,
                     "how many neurons in hidden layers of sampler")
flags.DEFINE_integer("noise_resolution", 64, "w, h, d of noise solid")
flags.DEFINE_integer("n_octaves", 16, "how many noise octaves to use. Needs to be either 16 or 32")
flags.DEFINE_integer("seed", -1, "seed for reproducibility, -1 for random")
flags.DEFINE_string("model_name", "./tf/chpts/run_00/checkpoint_s100000",
                    "checkpoint file to load file to load")
flags.DEFINE_string("out_path", './outputs/', 'path to output image(s)')

FLAGS = flags.FLAGS


def main(_):
    # setting seed for reproducibility
    if FLAGS.seed is not -1:
        tf.compat.v1.random.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        print("setting seed for reproducibility to " + str(FLAGS.seed))

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            # noise op
            s = FLAGS.noise_resolution
            noise_op = tf.random_normal([FLAGS.n_octaves, s, s, s], mean=0., stddev=1.)

            # placeholders
            slicing_matrix_ph = tf.placeholder(tf.float32, shape=[1, 4, 4])
            octaves_noise_ph = tf.placeholder(tf.float32, shape=[FLAGS.n_octaves, s, s, s])

            """
            TRANSFORMER: in single exemplar, we directly optimize for the transformation parameters
            """
            with tf.variable_scope('transformer'):
                transformations = tf.get_variable('octaves_transformations', shape=[FLAGS.n_octaves, 3, 3],
                                                  trainable=True, initializer=tf.random_normal_initializer())

            # broadcast to entire batch
            transformations = tf.expand_dims(transformations, 0)

            """
            SAMPLER
            """
            # single slice at z=0 [4, img_h*img_w]
            coords = meshgrid2D(FLAGS.img_h, FLAGS.img_w)
            # random slicing
            coords = tf.matmul(slicing_matrix_ph, coords)
            # drop homogeneous coordinate
            coords = coords[:, :3, :]

            S = models.sampler_single

            fake_img = S(octaves_noise_ph, coords, transformations, img_h=FLAGS.img_h, img_w=FLAGS.img_w,
                         act=tf.nn.leaky_relu, scope='sampler', hidden_dim=FLAGS.hidden_dim, eq_lr=True)

            # load model
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, FLAGS.model_name)
            print('loaded model from: ' + FLAGS.model_name)

            for i in range(FLAGS.num_samples):
                # get a random slicing matrices
                slicing_matrices = get_random_slicing_matrices(1)

                # get some noise
                noises = sess.run(noise_op)

                # get batch_size samples
                imgs = sess.run(fake_img, feed_dict={slicing_matrix_ph: slicing_matrices,
                                                     octaves_noise_ph: noises})
                img = imgs[0]
                cv2.imwrite(FLAGS.out_path + str(i+1) + '.png', cv2.cvtColor(np.clip(img, 0., 1.) * 255, cv2.COLOR_BGR2RGB),
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == "__main__":
    tf.app.run()
