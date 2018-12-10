import tensorflow as tf


def a2c_discrete_cnn(x):
    out = tf.cast(x, tf.float32) / 255.
    out = tf.layers.conv2d(out, 8, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 8, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2)
    out = tf.layers.conv2d(out, 16, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 16, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2)
    out = tf.layers.conv2d(out, 20, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 20, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2)

    out = tf.layers.dense(out, 100, activation=tf.nn.relu)
    out = tf.layers.dense(out, 1, activation=tf.nn.relu)

    return out


def a2c_cnn(x):
    out = tf.cast(x, tf.float32) / 255.
    out = tf.layers.conv2d(out, 8, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 8, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2)
    out = tf.layers.conv2d(out, 16, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 16, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2)
    out = tf.layers.conv2d(out, 20, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.conv2d(out, 20, 4, padding='SAME', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2)

    out = tf.layers.dense(out, 100, activation=tf.nn.relu)
    out = tf.layers.dense(out, 2, activation=tf.atan)

    return out
