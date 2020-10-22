# Copyright 2020 Tiziano Portenier
# Computer-assisted Applications in Medicine Group, Computer Vision Lab, ITET, ETH Zurich

import os
import tensorflow as tf
import numpy as np
import cv2

from utils import meshgrid2D, get_random_slicing_matrices
import models

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 16, "batch size")

flags.DEFINE_integer("hidden_dim", 128,
                     "how many neurons in hidden layers of sampler")

flags.DEFINE_float("beta", 1., "weight for optional (beta>0) style loss")
flags.DEFINE_float("alpha", .1, "weight for optional (alpha>0) GAN loss")

flags.DEFINE_integer("dis_iter", 1,
                     "how many discriminator updates per generator update")
flags.DEFINE_integer("progress_interval", 10000,
                     "dump weights every n steps")
flags.DEFINE_float("d_lr", 2e-3, "learning rate D")
flags.DEFINE_float("g_lr", 5e-4, "learning rate G")
flags.DEFINE_float("gp", 10, "gradient penalty weight")
flags.DEFINE_string("chpt_dir", "./tf/chpts",
                    "where to store checkpoints")
flags.DEFINE_string("sum_dir",  "./tf/summaries",
                    "where to store tensorboard summaries")

flags.DEFINE_integer("img_size", 128, "training image patch size")
flags.DEFINE_integer("noise_resolution", 64, "w, h, d of noise solid")
flags.DEFINE_integer("n_octaves", 16, "how many noise octaves to use. Needs to be either 16 or 32")
flags.DEFINE_string("training_exemplar",
                    './exemplars/0.png',
                    'training exemplar')
flags.DEFINE_bool("wood", False, "whether to use only specific slices, e.g., for wood, or completely random")

FLAGS = flags.FLAGS


def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = tf.concat(
            (tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])),
            axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(real, fake)
    pred, _ = f(x)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(
        tf.square(gradients),
        reduction_indices=range(1, x.shape.ndims)))
    gp = tf.reduce_mean((slopes - 1.) ** 2)

    return gp


def style_loss(feat_real, feat_fake):
    def gram_matrix(t):
        einsum = tf.linalg.einsum('bijc,bijd->bcd', t, t)
        n_pix = t.get_shape().as_list()[1]*t.get_shape().as_list()[2]
        return einsum/n_pix

    real_gram_mat = []
    fake_gram_mat = []

    # all D features except logits
    for i in range(len(feat_real)):
        real_gram_mat.append(gram_matrix(feat_real[i]))
        fake_gram_mat.append(gram_matrix(feat_fake[i]))

    # l1 loss
    style_loss = tf.add_n([tf.reduce_mean(tf.abs(real_gram_mat[idx] - fake_gram_mat[idx]))
                           for idx in range(len(feat_real))])

    return style_loss / len(feat_real)


def get_real_imgs(img):
    img_batch = []
    for i in range(FLAGS.batch_size):
        img_crop = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3])
        img_crop = tf.image.random_flip_left_right(img_crop)
        img_crop = tf.image.random_flip_up_down(img_crop)
        # add batch dimension
        img_batch.append(tf.expand_dims(img_crop, 0))
    img_batch = tf.concat(img_batch, 0)
    return img_batch


def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.device('/gpu:0'):
        with tf.Session(config=config) as sess:
            # load training exemplar
            training_img = cv2.imread(FLAGS.training_exemplar).astype(np.float32) / 255.
            training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2RGB)
            # drop alpha if present
            training_img = tf.convert_to_tensor(training_img[:, :, :3])

            real_batch_op = get_real_imgs(training_img)

            # noise op. Note that for efficiency, we use the same noise instance foreach sample in batch
            s = FLAGS.noise_resolution
            noise_op = tf.random_normal([FLAGS.n_octaves, s, s, s], mean=0., stddev=1.)

            # placeholders
            slicing_matrix_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 4, 4])
            octaves_noise_ph = tf.placeholder(tf.float32, [FLAGS.n_octaves, s, s, s])

            """
            TRANSFORMER: in single exemplar, we directly optimize for the transformation parameters
            """
            with tf.variable_scope('transformer'):
                transformations = tf.get_variable('octaves_transformations', shape=[FLAGS.n_octaves, 3, 3],
                                                  trainable=True, initializer=tf.random_normal_initializer())

            # broadcast to entire batch
            transformations = tf.expand_dims(transformations, 0)
            transformations = tf.tile(transformations, [FLAGS.batch_size, 1, 1, 1])

            """
            SAMPLER
            """
            # single slice at z=0 [4, img_size^2]
            coords = meshgrid2D(FLAGS.img_size, FLAGS.img_size)
            # bs different random slices [bs, img_size^2]
            coords = tf.matmul(slicing_matrix_ph, coords)
            # drop homogeneous coordinate
            coords = coords[:, :3, :]

            S = models.sampler_single

            fake_img = S(octaves_noise_ph, coords, transformations, img_size=FLAGS.img_size,
                         act=tf.nn.leaky_relu, scope='sampler', hidden_dim=FLAGS.hidden_dim, eq_lr=True)

            """
            DISCRIMINATOR
            """
            D = models.discriminator

            logits_fake, feat_fake = D(fake_img, reuse=False, scope='discriminator',
                                       act=tf.nn.leaky_relu, eq_lr=True)

            logits_real, feat_real = D(real_batch_op, reuse=True, scope='discriminator',
                                       act=tf.nn.leaky_relu, eq_lr=True)

            # D loss
            wgan_d_loss = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
            gp = gradient_penalty(real_batch_op, fake_img, D)
            d_loss = wgan_d_loss + FLAGS.gp * gp + 0.001 * logits_real**2

            # G loss
            g_style = (FLAGS.beta != 0.)
            g_gan = (FLAGS.alpha != 0)

            if g_style and g_gan:
                g_gan_loss = -tf.reduce_mean(logits_fake)
                g_style_loss = style_loss(feat_real, feat_fake)

            elif g_gan:
                g_gan_loss = -tf.reduce_mean(logits_fake)
                g_style_loss = 0.
            elif g_style:
                g_gan_loss = 0.
                g_style_loss = style_loss(feat_real, feat_fake)
            else:
                raise Exception("oops, must do either alpha or beta > 0!")

            g_loss = FLAGS.alpha * g_gan_loss + FLAGS.beta * g_style_loss

            # train steps
            d_vars = tf.trainable_variables(scope='discriminator')
            d_train_step = tf.train.AdamOptimizer(
                FLAGS.d_lr, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)

            g_vars = tf.trainable_variables(scope='transformer')
            g_vars += tf.trainable_variables(scope='sampler')
            g_train_step = tf.train.AdamOptimizer(
                FLAGS.g_lr, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)

            # summaries
            sum_img = tf.concat([real_batch_op, fake_img], 2)
            sum_img = tf.clip_by_value(sum_img, 0., 1.)
            g_summaries = [tf.summary.image('fake_img', sum_img, max_outputs=2)]
            if g_gan:
                g_summaries.append(tf.summary.scalar('g_gan_loss', g_gan_loss))
            if g_style:
                g_summaries.append(tf.summary.scalar('g_style_loss', g_style_loss))

            d_summaries = [tf.summary.scalar('logits_real', tf.reduce_mean(logits_real)),
                           tf.summary.scalar('logits_fake', tf.reduce_mean(logits_fake)),
                           tf.summary.scalar('d_loss', wgan_d_loss),
                           tf.summary.scalar('gp', gp)]

            # saver for chpts
            saver = tf.train.Saver(max_to_keep=None)

            # initialize variables
            print('')
            print("starting from scratch\n"
                  + "saving in " + FLAGS.chpt_dir)
            print('')
            sess.run(tf.initialize_all_variables())
            sum_writer = tf.summary.FileWriter(FLAGS.sum_dir, sess.graph)

            # training loop
            step = 0
            while True:
                # train D
                for i in range(FLAGS.dis_iter):
                    # get random slicing matrices
                    m = get_random_slicing_matrices(FLAGS.batch_size, wood=FLAGS.wood)
                    # get some noise
                    noise_input = sess.run(noise_op)

                    if step % 100 == 0 and i == 0:
                        run_results = sess.run(d_summaries + [d_train_step],
                                               feed_dict={slicing_matrix_ph: m,
                                                          octaves_noise_ph: noise_input})
                        # write summaries
                        for s in run_results[:len(d_summaries)]:
                            sum_writer.add_summary(s, step)

                    else:
                        sess.run(d_train_step,
                                 feed_dict={slicing_matrix_ph: m,
                                            octaves_noise_ph: noise_input})
                # train G
                if step % 100 == 0:
                    run_results = sess.run(g_summaries + [g_train_step],
                                           feed_dict={slicing_matrix_ph: m,
                                                      octaves_noise_ph: noise_input})
                    for s in run_results[:len(g_summaries)]:
                        sum_writer.add_summary(s, step)
                else:
                    sess.run(g_train_step, feed_dict={slicing_matrix_ph: m,
                                                      octaves_noise_ph: noise_input})
                step += 1

                # store chpt
                if step % FLAGS.progress_interval == 0:
                    saver.save(sess, FLAGS.chpt_dir + '/checkpoint_s' + str(step))


def main(_):
    # create directories for data and summaries if necessary
    if not os.path.exists(FLAGS.chpt_dir):
        FLAGS.chpt_dir += '/run_00'
        os.makedirs(FLAGS.chpt_dir)
    else:
        # find last run number
        dirs = next(os.walk(FLAGS.chpt_dir))[1]
        for idx, d in enumerate(dirs):
            dirs[idx] = d[-2:]

        runs = sorted(map(int, dirs))
        run_nr = 0
        if len(runs) > 0:
            run_nr = runs[-1] + 1

        FLAGS.chpt_dir += '/run_' + str(run_nr).zfill(2)
        os.makedirs(FLAGS.chpt_dir)

    if not os.path.exists(FLAGS.sum_dir):
        FLAGS.sum_dir += '/run_00'
        os.makedirs(FLAGS.sum_dir)
    else:
        # find last run number
        dirs = next(os.walk(FLAGS.sum_dir))[1]
        for idx, d in enumerate(dirs):
            dirs[idx] = d[-2:]

        runs = sorted(map(int, dirs))
        run_nr = 0
        if len(runs) > 0:
            run_nr = runs[-1] + 1

        FLAGS.sum_dir += '/run_' + str(run_nr).zfill(
            2)
        os.makedirs(FLAGS.sum_dir)

    # train
    train()


if __name__ == "__main__":
    tf.app.run()
