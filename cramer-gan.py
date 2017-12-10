import os
import time
import argparse
import importlib
import tensorflow as tf
from scipy.misc import imsave
from visualize import *


def _safer_norm(tensor, axis=None, keep_dims=False, epsilon=1e-5):
  sq = tf.square(tensor)
  squares = tf.reduce_sum(sq, axis=axis, keep_dims=keep_dims)
  return tf.sqrt(squares + epsilon)


def avg_distance(diff):
  diff.shape.assert_is_compatible_with([None, None])
  return tf.reduce_mean(_safer_norm(diff, axis=-1))


def _gp_critic(h_interpolates,
               h_real,
               h_generated):
  # The critic is:
  #   f(h) = |h - h_generated| - |h - h_real|
  h_interpolates.shape.assert_is_compatible_with([None, None])
  h_interpolates.shape.assert_is_compatible_with(h_real.shape)
  h_interpolates.shape.assert_is_compatible_with(h_generated.shape)
  return tf.add_n([
      _safer_norm(h_interpolates - h_generated, axis=-1, keep_dims=True),
      -_safer_norm(h_interpolates - h_real, axis=-1, keep_dims=True),
  ])


def _compute_surrogate_loss(
    d_real_to_generated1,
    d_real_to_generated2,
    d_generated1_to_generated2,
    d_real_to_0,
    d_generated1_to_0,
    d_generated2_to_0):
  # The surrogate loss is:
  #   surrogate_generator_loss = (
  #       0.5 * |h_real - h_generated1|
  #       0.5 * |h_real - h_generated2|
  #       - |h_real|
  #       - |h_generated1 - h_generated2|
  #       + 0.5 * |h_generated1|
  #       + 0.5 * |h_generated2|)
  return tf.add_n([
      0.5 * d_real_to_generated1,
      0.5 * d_real_to_generated2,
      -d_real_to_0,
      -d_generated1_to_generated2,
      0.5 * d_generated1_to_0,
      0.5 * d_generated2_to_0,
  ])


class CramerGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model, scale=10.0):
        self.model = model
        self.data = data
        self.d_net = d_net
        self.g_net = g_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z1 = tf.placeholder(tf.float32, [None, self.z_dim], name='z1')
        self.z2 = tf.placeholder(tf.float32, [None, self.z_dim], name='z2')

        self.x1_ = self.g_net(self.z1, reuse=False)
        self.x2_ = self.g_net(self.z2)

        h_real = d_net(self.x, reuse=False)
        h_generated1 = d_net(self.x1_)
        h_generated2 = d_net(self.x2_)

        # If having independent examples in the minibatch,
        # it would be possible to construct `batch_size * (batch_size - 1)`
        # independent pairs.
        # Here, we construct just `batch_size` independent pairs
        # to be able to use the same code for conditional modeling.
        d_real_to_generated1 = avg_distance(h_real - h_generated1)
        d_real_to_generated2 = avg_distance(h_real - h_generated2)
        d_generated1_to_generated2 = avg_distance(h_generated1 - h_generated2)
        d_real_to_0 = avg_distance(h_real)
        d_generated1_to_0 = avg_distance(h_generated1)
        d_generated2_to_0 = avg_distance(h_generated2)

        # The energy_g_loss is the energy distance without
        # the |h_real - h'_real| term.
        energy_g_loss = tf.add_n([
            d_real_to_generated1,
            d_real_to_generated2,
            -d_generated1_to_generated2,
        ])
        # surrogate generator loss
        surrogate_g_loss = _compute_surrogate_loss(
            d_real_to_generated1=d_real_to_generated1,
            d_real_to_generated2=d_real_to_generated2,
            d_generated1_to_generated2=d_generated1_to_generated2,
            d_real_to_0=d_real_to_0,
            d_generated1_to_0=d_generated1_to_0,
            d_generated2_to_0=d_generated2_to_0)

        self.g_loss = energy_g_loss
        self.d_loss = -surrogate_g_loss

        # interpolate real and generated samples
        epsilon = tf.random_uniform([], 0.0, 1.0)
        # Using x and x1_ for the x_hat
        # and using the corresponding h_real and h_generated1
        # in the _gp_critic.
        x_hat = epsilon * self.x + (1 - epsilon) * self.x1_
        h_interpolates = d_net(x_hat)
        d_hat = _gp_critic(
            h_interpolates=h_interpolates,
            h_real=h_real,
            h_generated=h_generated1)

        ddx = tf.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = tf.norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            #if t % 500 == 0 or t < 25:
            #     d_iters = 100

            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz1 = self.z_sampler(batch_size, self.z_dim)
                bz2 = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z1: bz1, self.z2: bz2})

            bx = self.x_sampler(batch_size)
            bz1 = self.z_sampler(batch_size, self.z_dim)
            bz2 = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_adam, feed_dict={self.z1: bz1, self.x: bx, self.z2: bz2})

            if t % 100 == 0:
                bx = self.x_sampler(batch_size)
                bz1 = self.z_sampler(batch_size, self.z_dim)
                bz2 = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z1: bz1, self.z2: bz2}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z1: bz1, self.z2: bz2, self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss, g_loss))

            if t % 100 == 0:
                bz1 = self.z_sampler(batch_size, self.z_dim)
                bx = self.sess.run(self.x1_, feed_dict={self.z1: bz1})
                bx = xs.data2img(bx)
                bx = grid_transform(bx, xs.shape)
                imsave('logs/{}/{}.png'.format(self.data, t/100), bx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    try:
        os.makedirs('logs/{}'.format(args.data))
    except Exception:
        print('logs/{}'.format(args.data) + 'not created')
        pass
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    cgan = CramerGAN(g_net, d_net, xs, zs, args.data, args.model)
    cgan.train()
