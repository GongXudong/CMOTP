import numpy as np
import tensorflow as tf
def play():

    raise NotImplementedError()


if __name__ == '__main__':
    a = tf.placeholder(tf.int32, shape=(None, 1), name='a')

    b = tf.cond(a > 0, tf.greater())




if __name__ == '__main1__':
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(c))