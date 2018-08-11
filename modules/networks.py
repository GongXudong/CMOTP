
import numpy as np
import tensorflow as tf
import math

class QNetwork(object):
    """
	Base class for QNetworks.
	"""

    def __init__(self, input_size, output_size, name):
        self.name = name

    def weight_variable(self, shape, name, fanin=0):
        if fanin == 0:
            initial = tf.truncated_normal(shape, stddev=0.01)
        else:
            mod_init = 1.0 / math.sqrt(fanin)
            initial = tf.random_uniform(shape, minval=-mod_init, maxval=mod_init)

        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name, fanin=0):
        if fanin == 0:
            initial = tf.constant(0.01, shape=shape)
        else:
            mod_init = 1.0 / math.sqrt(fanin)
            initial = tf.random_uniform(shape, minval=-mod_init, maxval=mod_init)

        return tf.Variable(initial, name=name)

    def variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def copy_to(self, dst_net):
        v1 = self.variables()
        v2 = dst_net.variables()

        for i in range(len(v1)):
            v2[i].assign(v1[i]).eval()

    def print_num_of_parameters(self):
        list_vars = self.variables()
        total_parameters = 0
        for variable in list_vars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('# of parameters in network ', self.name, ': ', total_parameters, '  ->  ',
              np.round(float(total_parameters) / 1000000.0, 3), 'M')


class QNetworkMLP(QNetwork):

    def __init__(self, name, input_size, output_size):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size

        with tf.variable_scope(self.name):

            # FC layer 1
            self.W_fc1 = self.weight_variable([input_size, 128], name='W1')  # , fanin=11*11*32)
            self.B_fc1 = self.bias_variable([128], name = 'B1')  # , fanin=11*11*32)

            # FC layer 2
            self.W_fc2 = self.weight_variable([128, 64], name='W2')  # , fanin=256)
            self.B_fc2 = self.bias_variable([64], 'B2')  # , fanin=256)

            # FC layer 3
            self.W_fc3 = self.weight_variable([64, self.output_size], name='W3')  # , fanin=256)
            self.B_fc3 = self.bias_variable([self.output_size], name='B3')  # , fanin=256)

        # Print number of parameters in the network
        self.print_num_of_parameters()

    def __call__(self, input_tensor):
        if type(input_tensor) == list:
            input_tensor = tf.concat(1, input_tensor)
        with tf.variable_scope(self.name):

            self.h_fc1 = tf.nn.relu(tf.matmul(input_tensor, self.W_fc1) + self.B_fc1)
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.B_fc2)
            self.h_fc3 = tf.identity(tf.matmul(self.h_fc2, self.W_fc3) + self.B_fc3)

        return self.h_fc3


if __name__ == "__main__":
    QNetworkMLP('cmotp', 6,  10)
