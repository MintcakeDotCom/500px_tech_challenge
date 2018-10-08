import tensorflow as tf


def weight(kernel_shape, is_uniform=True):

    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w


def bias(bias_shape):

    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b


def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def deconv(layer_name, x, out_channels, output_shape=[32, 224, 224, 64], kernel_size=[3, 3], stride=[1, 1, 1, 1]):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1], out_channels, in_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        # b = tf.get_variable(name='biases',
        #                     shape=[out_channels],
        #                     initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='SAME',
                                   name='deconv')
        # x = tf.nn.bias_add(x, b, name='bias_add')
        # x = tf.nn.relu(x, name='relu')
        return x


def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
    x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


def FC_layer(layer_name, x, out_nodes):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))

        # flatten input into 1-dimension, achieve the same function of tf.keras.layers.Flatten
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        # x = tf.nn.relu(x)

        return x


def drop(x, prob):
    x = tf.nn.dropout(x, keep_prob=prob)

    return x

