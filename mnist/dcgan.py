import tensorflow as tf

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(x * alpha, 0), x)


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tf.layers.conv2d(x, 64, [4, 4], [2, 2])
            conv1 = leaky_relu(conv1)
            conv2 = tf.layers.conv2d(conv1, 128, [4, 4], [2, 2])
            conv2 = leaky_relu(conv2)
            conv2 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.layers.dense(conv2, 1024)
            fc1 = leaky_relu(fc1)
            fc2 = tf.layers.dense(fc1, 256)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784
        self.name = 'mnist/dcgan/g_net'

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            bs = tf.shape(z)[0]
            fc1 = tf.layers.dense(z, 1024)
            fc1 = tf.nn.relu(fc1)
            fc2 = tf.layers.dense(fc1, 7 * 7 * 128)
            fc2 = tf.reshape(fc2, tf.stack([bs, 7, 7, 128]))
            fc2 = tf.nn.relu(fc2)
            conv1 = tf.contrib.layers.conv2d_transpose(fc2, 64, [4, 4], [2, 2])
            conv1 = tf.nn.relu(conv1)
            conv2 = tf.contrib.layers.conv2d_transpose(conv1, 1, [4, 4], [2, 2], activation_fn=tf.sigmoid)
            conv2 = tf.reshape(conv2, tf.stack([bs, 784]))
            return conv2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]