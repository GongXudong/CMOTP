
import time
import logging
import tensorflow as tf
import numpy as np
from env.cmotp import CMOTP
from modules.ddqn import DDQN
import sys

# 每一局游戏最多进行多少step
MAX_GAME_STEPS = 500
# 总共训练多少step
TOTAL_STEPS = 1000000


# 多少个step之后开始训练
TRAIN_AFTER_STEPS = 10000

# 每隔多少step测试一次
TEST_AVERAGE_STEPS = 1000
# 一次测试运行多少局游戏
TESTING_GAMES = 30
# 测试中，每一局游戏最多运行多少step
MAX_TESTING_STEPS = 100

DEBUG = True

env = CMOTP()
env_test = CMOTP()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.InteractiveSession(config=config)


agent = DDQN(state_size=6,
             action_size=25,
             exploration_period=100000,
             minibatch_size=128,
             discount_factor=0.98,
             experience_replay_buffer=20000,
             target_qnet_update_frequency=1000,
             save_frequency=5000,
             initial_exploration_epsilon=1.0,
             final_exploration_epsilon=0.05)

test_mode = False
if len(sys.argv) >= 2:
    agent.saver.restore(agent.session, sys.argv[1])
if len(sys.argv) == 3:
    test_mode = sys.argv[2] == 'True'

# 记录当前正在进行第几个step
num_steps = 0
# 记录当前游戏正在进行第几个step
current_game_steps = 0
# 记录总共进行了多少局游戏
num_games = 0

state = env.reset()

while num_steps < TOTAL_STEPS:

    step_begin_time = time.time()

    num_steps += 1
    current_game_steps += 1

    if not test_mode:
        action = agent.action(np.array(state[0] + state[1]), training=True)
    else:
        action = agent.action(np.array(state[0] + state[1]), training=False)

    action_1 = int(action / 5)
    action_2 = action % 5
    action_n = [action_1, action_2]

    next_state, reward, done, _ = env.step(action_n)
    if DEBUG:
        if num_steps > TRAIN_AFTER_STEPS:
            print('cur_state:', state, 'action: ', action_n, 'next_state: ', next_state, 'reward: ', reward)

    agent.store(np.array(state[0] + state[1]), action, reward[0], np.array(next_state[0] + next_state[1]), done[0])

    if DEBUG:
        if num_steps > TRAIN_AFTER_STEPS:
            print(num_steps)
        # print(num_steps, state, action_n, next_state)
        if reward[0] > 0.:
            print('got a reward:', reward[0])
            if reward[0] > 5.:
                time.sleep(5)
            print(next_state)

    state = next_state



    train_begin_time = time.time()
    if num_steps > TRAIN_AFTER_STEPS:
        agent.train()
    train_end_time = time.time()

    if done[0] or current_game_steps > MAX_GAME_STEPS:
        if done[0]:
            print('success!')
        if current_game_steps > MAX_GAME_STEPS:
            print('exceed...')
        state = env.reset()
        current_game_steps = 0
        num_games += 1

    step_end_time = time.time()
    if DEBUG:
        if num_steps > TRAIN_AFTER_STEPS:
            print('time cost, train:{}'.format((train_end_time - train_begin_time)/(step_end_time - step_begin_time)))

    # test
    if num_steps % TEST_AVERAGE_STEPS == 0 and num_steps > TRAIN_AFTER_STEPS:
        total_reward = 0.
        total_steps = 0

        for i in range(TESTING_GAMES):
            state_in_test = env_test.reset()
            cur_steps_in_test = 0
            while cur_steps_in_test < MAX_TESTING_STEPS:
                cur_steps_in_test += 1
                action_in_test = agent.action(np.array(state_in_test[0] + state_in_test[1]), training=False)
                action_1_in_test = int(action_in_test / 5)
                action_2_in_test = action_in_test % 5
                action_n_in_test = [action_1_in_test, action_2_in_test]
                if i == 1:
                    if cur_steps_in_test == 1:
                        print('test1')
                        agent.action(np.array((5, 0, 1, 5, 4, 1)), training=False)
                        print('test2')
                        agent.action(np.array((4, 2, 1, 5, 2, 1)), training=False)
                        print('test3')
                        agent.action(np.array((3, 2, 1, 3, 4, 1)), training=False)
                        print('test3')
                        agent.action(np.array((2, 2, 2, 2, 4, 3)), training=False)
                        print('test4')
                        agent.action(np.array((0, 4, 2, 0, 6, 3)), training=False)


                    print("state: ", state_in_test)
                    print("action:", action_in_test)
                    print("action_n:", action_n_in_test)
                    env_test.render()


                state_in_test, reward_in_test, done_in_test, _ = env_test.step(action_n_in_test)

                total_reward += reward_in_test[0]
                if done_in_test[0]:
                    break

            total_steps += cur_steps_in_test
        str_ = agent.session.run(tf.summary.scalar('test reward (' + str(num_steps / 1000) + 'k)',
                                             float(total_reward)/TESTING_GAMES))
        agent.summary_writer.add_summary(str_, num_steps)

        print('  --> Evaluation Average Reward: ', float(total_reward)/TESTING_GAMES,
              '   avg steps: ', (total_steps / TESTING_GAMES))



agent.summary_writer.close()

agent.saver.save(agent.session, 'save/model_final')

