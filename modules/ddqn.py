
import numpy as np
import tensorflow as tf
import random
from modules.networks import QNetworkMLP
from modules.ReplayBuffer import ReplayBuffer

class DDQN(object):

    def __init__(self,
                 state_size,
                 action_size,
                 exploration_period=10000,
                 minibatch_size=32,
                 discount_factor=0.99,
                 experience_replay_buffer=20000,
                 target_qnet_update_frequency=10000,
                 save_frequency=1000,
                 initial_exploration_epsilon=1.0,
                 final_exploration_epsilon=0.05,
                 reward_clipping=-1,
                 ):

        # Setup the parameters, data structures and networks
        self.state_size = state_size
        self.action_size = action_size
        self.exploration_period = float(exploration_period)
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.experience_replay_buffer = experience_replay_buffer
        self.reward_clipping = reward_clipping

        self.target_qnet_update_frequency = target_qnet_update_frequency
        self.save_frequency = save_frequency
        self.initial_exploration_epsilon = initial_exploration_epsilon
        self.final_exploration_epsilon = final_exploration_epsilon

        self.qnet = QNetworkMLP("qnet", self.state_size, self.action_size)
        self.target_qnet = QNetworkMLP("target_qnet", self.state_size, self.action_size)

        # self.qnet_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, epsilon=0.01)
        self.qnet_optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

        self.experience_replay = ReplayBuffer(self.experience_replay_buffer)

        self.num_training_steps = 0

        # Setup the computational graph
        self.create_graph()

        LOG_DIR = 'results/'
        self.summary_writer = tf.summary.FileWriter(LOG_DIR)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(tf.global_variables())

        self.session.run(tf.global_variables_initializer())
        self.summary_writer.add_graph(self.session.graph)

        checkpoint = tf.train.get_checkpoint_state("save")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    @staticmethod
    def copy_to_target_network(source_network, target_network):
        target_network_update = []
        for v_source, v_target in zip(source_network.variables(), target_network.variables()):
            # this is equivalent to target = source
            update_op = v_target.assign(v_source)
            target_network_update.append(update_op)
        return tf.group(*target_network_update)


    def create_graph(self):
        # Pick action given state ->   action = argmax( qnet(state) )
        with tf.name_scope("pick_action"):
            self.state = tf.placeholder(tf.float32, (None, self.state_size), name="state")

            # self.q_values = tf.identity(self.qnet(self.state), name="q_values")
            self.q_values = self.qnet(self.state)
            self.predicted_actions = tf.argmax(self.q_values, axis=1, name="predicted_actions")

            tf.summary.histogram("Q values",
                                 tf.reduce_mean(tf.reduce_max(self.q_values, 1)))  # save max q-values to track learning

        # Predict target future reward: r  +  gamma * max_a'[ Q'(s') ]
        # self.target_q_values = self.rewards + self.discount_factor * self.next_max_q_values
        self.target_q_values = tf.placeholder(tf.float32, (None, ), name="target_q_values")

        # Gradient descent
        with tf.name_scope("optimization_step"):
            self.action_mask = tf.placeholder(tf.float32, (None, self.action_size),
                                              name="action_mask")  # action that was selected
            self.y = tf.reduce_sum(self.q_values * self.action_mask, axis=1)

            ## ERROR CLIPPING AS IN NATURE'S PAPER
            self.error = tf.abs(self.y - self.target_q_values)
            quadratic_part = tf.clip_by_value(self.error, 0.0, 1.0)
            linear_part = self.error - quadratic_part
            self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            qnet_gradients = self.qnet_optimizer.compute_gradients(self.loss, self.qnet.variables())
            for i, (grad, var) in enumerate(qnet_gradients):
                if grad is not None:
                    qnet_gradients[i] = (tf.clip_by_norm(grad, 10), var)
            self.qnet_optimize = self.qnet_optimizer.apply_gradients(qnet_gradients)

        with tf.name_scope("target_network_update"):
            self.hard_copy_to_target = DDQN.copy_to_target_network(self.qnet, self.target_qnet)

        self.summarize = tf.summary.merge_all()



    def store(self, state, action, reward, next_state, is_terminal):
        # rewards clipping
        if self.reward_clipping > 0.0:
            reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)

        self.experience_replay.add(state, action, reward, next_state, is_terminal)

    def action(self, state, training=False):
        """
                If `training', compute the epsilon-greedy parameter epsilon according to the defined exploration_period, initial_epsilon and final_epsilon.
                If not `training', use a fixed testing epsilon=0.05
                """
        if self.num_training_steps > self.exploration_period:
            epsilon = self.final_exploration_epsilon
        else:
            epsilon = self.initial_exploration_epsilon - float(self.num_training_steps) * (
                    self.initial_exploration_epsilon - self.final_exploration_epsilon) / self.exploration_period

        if not training:
            epsilon = 0.05

        # Execute a random action with probability epsilon, or follow the QNet policy with probability 1-epsilon.
        if random.random() <= epsilon:
            # print('exploration, ', epsilon)
            action = random.randint(0, self.action_size - 1)
        else:
            # print('max, ', epsilon)
            q_value = self.session.run(self.q_values, {self.state: [state]})
            print(q_value)
            action = self.session.run(self.predicted_actions, {self.state: [state]})[0]
        return action

    def train(self):

        self.num_training_steps += 1

        # Copy the QNetwork weights to the Target QNetwork.
        if self.num_training_steps == 0:
            print("Training starts...")
            self.session.run(self.hard_copy_to_target)

        # Sample experience from replay memory
        batch_states, actions, batch_rewards, batch_newstates, batch_newstates_mask = self.experience_replay.sample(self.minibatch_size)
        batch_states = np.float32(batch_states)
        batch_newstates = np.float32(batch_newstates)
        if len(batch_states) == 0:
            return

        batch_actions = np.zeros((self.minibatch_size, self.action_size))
        for i in range(self.minibatch_size):
            batch_actions[i, actions[i]] = 1


        next_q_values_targetqnet = self.session.run(self.target_qnet(batch_newstates))

        next_q_values_qnet = self.session.run(self.qnet(batch_newstates))
        next_selected_actions = np.argmax(next_q_values_qnet, axis=1)
        next_selected_actions_onehot = np.zeros((self.minibatch_size, self.action_size))
        for i in range(self.minibatch_size):
            next_selected_actions_onehot[i, next_selected_actions[i]] = 1

        next_max_q_values = np.sum(next_q_values_targetqnet * next_selected_actions_onehot, axis=1)

        target_q_values = batch_rewards + self.discount_factor * next_max_q_values * (1 - batch_newstates_mask)

        # Perform training
        scores, _, = self.session.run([self.q_values, self.qnet_optimize],
                                      {self.state: batch_states,
                                       self.action_mask: batch_actions,
                                       self.target_q_values: target_q_values})


        if self.num_training_steps % self.target_qnet_update_frequency == 0:
            # Hard update (copy) of the weights every # iterations
            self.session.run(self.hard_copy_to_target)

            # Write logs
            print('mean maxQ in minibatch: ', np.mean(np.max(scores, 1)))

            str_ = self.session.run(self.summarize, {self.state: batch_states,
                                                     self.action_mask: batch_actions,
                                                     self.target_q_values: target_q_values})
            self.summary_writer.add_summary(str_, self.num_training_steps)

        if self.num_training_steps % self.save_frequency == 0:
            self.saver.save(self.session, 'save/model', global_step=self.num_training_steps)






if __name__ == '__main__':

    minibatch_size = 2
    action_size = 3
    discount_factor = 2
    next_q_values_targetqnet = np.array([[1, 2, 3],
                                         [2, 5, 1]])

    next_q_values_qnet = np.array([[3, 2, 1],
                                   [1, 4, 3]])
    batch_rewards = np.array([2, 3])
    batch_newstates_mask = np.array([False, True])
    next_selected_actions = np.argmax(next_q_values_qnet, axis=1)
    next_selected_actions_onehot = np.zeros((minibatch_size, action_size))
    for iii in range(minibatch_size):
        next_selected_actions_onehot[iii, next_selected_actions[iii]] = 1

    next_max_q_values = np.sum(next_q_values_targetqnet * next_selected_actions_onehot, axis=1)

    target_q_values = batch_rewards + discount_factor * next_max_q_values * (1 - batch_newstates_mask)

    print(target_q_values)