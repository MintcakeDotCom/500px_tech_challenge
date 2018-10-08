import tensorflow as tf
import tools


def mnist_net(x, prob):
    with tf.variable_scope('mnist_net'):

        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # first convolution layer
        x = tools.conv('conv_layer_1', x, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1])
        with tf.name_scope('max_pool_1'):
            x = tools.pool('max_pool_1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # second convolution layer
        x = tools.conv('conv_layer_2', x, 64, kernel_size=[5, 5], stride=[1, 1, 1, 1])
        with tf.name_scope('max_pool_2'):
            x = tools.pool('max_pool_2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # fully convolution layers
        x = tools.FC_layer('fc_layer_1', x, 1024)
        x = tf.nn.relu(x)
        x = tools.drop(x, prob)
        x = tools.FC_layer('fc_layer_2', x, 10)

        return x


# generator
def Generator(x):
    with tf.variable_scope('Generater'):

        x = tools.FC_layer('fc_gen', x, 7*7*64)

        x = tf.nn.relu(x)
        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = tools.deconv('deconv_1_gen', x, 32, output_shape=[50, 14, 14, 32], kernel_size=[5, 5], stride=[1, 2, 2, 1])
        x = tf.nn.relu(x)

        x_logits = tools.deconv('deconv_2_gen', x, 1, output_shape=[50, 28, 28, 1], kernel_size=[5, 5], stride=[1, 2, 2, 1])
        x = tf.nn.tanh(x_logits)

        return x_logits, x

