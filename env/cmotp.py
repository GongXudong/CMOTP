import gym
from enum import Enum

class CMOTP(gym.Env):

    GRASP_STATE = Enum('GRASP_STATE', ('FREE', 'GRASPING_LEFT', 'GRASPING_RIGHT'))

    def __init__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None

    def reset(self):
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [-1, -1, 0, -1, -1, -1, -1],
                    [1, 0, 0, 0, 2, 0, -1]]
        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region = [(0, 2), (0, 3), (0, 4)]
        return (5, 0, self.GRASP_STATE.FREE.value), (5, 4, self.GRASP_STATE.FREE.value)

    def step(self, action_n):
        # validate actions
        for i, ac in enumerate(action_n):
            assert ac >= 0 and ac <= 4, "agent {}'s action is out of range.".format(i+1)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()
        grasp_before_move = self.grasp_goods(pos_agent1, pos_agent2, pos_goods)
        self.move(action_n)
        new_pos_agent1, new_pos_agent2, new_pos_goods = self.get_entity_pos()
        grasp_after_move = self.grasp_goods(new_pos_agent1, new_pos_agent2, new_pos_goods)

        agent1_grasp_state = self.GRASP_STATE.FREE
        agent2_grasp_state = self.GRASP_STATE.FREE

        if grasp_after_move:
            # 暂只考虑左右抓取货物的情况
            if new_pos_agent1[1] + 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_LEFT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
            elif new_pos_agent1[1] - 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_LEFT

        terminate = False

        reward = 0.

        if not grasp_before_move and grasp_after_move:
            reward = 1.
        if self.home_region.__contains__(new_pos_goods):
            reward = 10.
            terminate = True
        # return (self.map, self.map), (reward, reward), (terminate, terminate), ({}, {})
        return ((*new_pos_agent1, agent1_grasp_state.value), (*new_pos_agent2, agent2_grasp_state.value)), \
               (reward, reward), (terminate, terminate), ({}, {})

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        grid_len = 50.
        screen_width = grid_len * self.map_size[1]
        screen_height = grid_len * self.map_size[0]

        def map_coor(pos):
            """
            self.map的坐标映射为self.viewer中的坐标
            :param pos: 横坐标与纵坐标构成的tuple
            :return:
            """
            return (pos[1] + 0.5) * grid_len, (self.map_size[0] - 1 - pos[0] + 0.5) * grid_len

        def make_rectangle(pos):
            """
            给定格子的坐标，生成画布中的四边形的4个顶点坐标
            :param pos: 横坐标与纵坐标构成的tuple
            :return:
            """
            center_x, center_y = map_coor(pos)
            return [(center_x - grid_len / 2, center_y - grid_len / 2),
                    (center_x - grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y - grid_len / 2)]

        if self.viewer == None:
            self.viewer = rendering.Viewer(int(screen_width), int(screen_height))

            # draw lines
            lines_view = []
            lines_view.append(rendering.Line((0., 0.), (screen_width, 0.)))
            lines_view.append(rendering.Line((0., 0.), (0., screen_height)))

            for i in range(self.map_size[0]):
                lines_view.append(rendering.Line((0., (i + 1) * grid_len), (screen_width, (i + 1) * grid_len)))

            for i in range(self.map_size[1]):
                lines_view.append(rendering.Line(((i + 1) * grid_len, 0.), ((i + 1) * grid_len, screen_height)))

            for i in range(len(lines_view)):
                lines_view[i].set_color(0, 0, 0)

            # draw obstacles
            obstacles_view = []
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if self.map[i][j] == -1:
                        obstcl = rendering.make_polygon(make_rectangle((i, j)))
                        obstcl.set_color(0, 0, 0)
                        obstacles_view.append(obstcl)
            home_regions_view = []
            for home in self.home_region:
                home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
                home_view.set_color(100, 100, 0)
                home_regions_view.append(home_view)
            # draw agents and goods

            agent1_view = rendering.make_circle(20)
            self.agent1_trans = rendering.Transform()
            agent1_view.add_attr(self.agent1_trans)
            agent1_view.set_color(1, 0, 0)

            agent2_view = rendering.make_circle(20)
            self.agent2_trans = rendering.Transform()
            agent2_view.add_attr(self.agent2_trans)
            agent2_view.set_color(0, 1, 0)

            goods_view = rendering.make_circle(15)
            self.goods_trans = rendering.Transform()
            goods_view.add_attr(self.goods_trans)
            goods_view.set_color(0, 0, 1)

            for i in range(len(lines_view)):
                self.viewer.add_geom(lines_view[i])
            self.viewer.add_geom(agent1_view)
            self.viewer.add_geom(agent2_view)
            self.viewer.add_geom(goods_view)
            for i in obstacles_view:
                self.viewer.add_geom(i)
            for i in home_regions_view:
                self.viewer.add_geom(i)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()

        self.agent1_trans.set_translation(*map_coor(pos_agent1))
        self.agent2_trans.set_translation(*map_coor(pos_agent2))
        self.goods_trans.set_translation(*map_coor(pos_goods))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def get_entity_pos(self):
        """
        返回agents的位置与goods的位置
        :return:
        """
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j] == 1:
                    pos_agent1 = (i, j)
                elif self.map[i][j] == 2:
                    pos_agent2 = (i, j)
                elif self.map[i][j] == 3:
                    pos_goods = (i, j)
        return pos_agent1, pos_agent2, pos_goods

    def grasp_goods(self, pos_agent1, pos_agent2, pos_goods):
        """
        判断两个agent是否“水平”运载着goods
        :param pos_agent1: agent1的位置
        :param pos_agent2: agent2的位置
        :param pos_goods: goods的位置
        :return: 运载，则返回True，否则返回False
        """

        def judge(_a, _b, _c):
            """
            判断_a, _b, _c是否构成等差数列，公差为+1或-1
            :param _a:
            :param _b:
            :param _c:
            :return:
            """
            if _b - _a == _c - _b and abs(_c - _b) == 1:
                return True
            return False

        if pos_agent1[0] == pos_agent2[0] and pos_agent2[0] == pos_goods[0]:
            return judge(pos_agent1[1], pos_goods[1], pos_agent2[1])

        # 只能左右抓取货物，所以下面2行代码暂时不要
        # if pos_agent1[1] == pos_agent2[1] and pos_agent2[1] == pos_goods[1]:
        #     return judge(pos_agent1[0], pos_goods[0], pos_agent2[0])

        return False

    def move(self, action_n):
        """
        规则：如果agent要移动到的格子不为空，则这个agent不动；如果两个agent要移动到同一个格子，则这两个agent都不动
        :param action_n:
        :return:
        """
        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()
        if self.grasp_goods(pos_agent1, pos_agent2, pos_goods):
            self.move_grasp(pos_agent1, pos_agent2, pos_goods, action_n)
        else:
            self.move_not_grasp(pos_agent1, pos_agent2, action_n)

    def move_not_grasp(self, pos_agent1, pos_agent2, action_n):
        if action_n[0] == 0 or action_n[1] == 0:
            # 至少一个agent的动作是“不动”的情况
            if action_n[0] == 0 and action_n[1] == 0:
                # 两个都不动的情况
                return
            if action_n[0] == 0:
                # agent1不动的情况
                new_pos_agent2 = (pos_agent2[0] + self.move_delta[action_n[1]][0],
                                  pos_agent2[1] + self.move_delta[action_n[1]][1])
                valid_state2 = self.valid(new_pos_agent2)
                if valid_state2 == 0 or valid_state2 == 2:
                    # agent2要移动向障碍或者goods或者agent1，则agent2原地不动
                    return
                else:
                    # agent2要移动向空格
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
            if action_n[1] == 0:
                # agent2不动的情况
                new_pos_agent1 = (pos_agent1[0] + self.move_delta[action_n[0]][0],
                                  pos_agent1[1] + self.move_delta[action_n[0]][1])
                valid_state1 = self.valid(new_pos_agent1)
                if valid_state1 == 0 or valid_state1 == 2:
                    # agent1要移动向障碍或者goods或者agent2，则agent1原地不动
                    return
                else:
                    # agent1要移动向空格
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
        else:
            # 两个agent动作都不是“不动”的情况
            new_pos_agent1 = (pos_agent1[0] + self.move_delta[action_n[0]][0],
                              pos_agent1[1] + self.move_delta[action_n[0]][1])
            new_pos_agent2 = (pos_agent2[0] + self.move_delta[action_n[1]][0],
                              pos_agent2[1] + self.move_delta[action_n[1]][1])
            valid_state1 = self.valid(new_pos_agent1)
            valid_state2 = self.valid(new_pos_agent2)
            if new_pos_agent1 == new_pos_agent2:
                # 两个agent要移向同一个位置
                return
            if valid_state1 == 0:
                # agent1不能移动的情况
                if valid_state2 == 2:
                    # agent2要移动向agent1，此时agent2也不能动
                    return
                elif valid_state2 == 0:
                    # agent2也不能移动
                    return
                else:
                    # agent2移动向空格
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
            elif valid_state1 == 2:
                # agent1要移动向agent2的情况
                if valid_state2 == 0:
                    # agent2不能移动的情况
                    return
                elif valid_state2 == 2:
                    # agent2要移动向agent1
                    return
                else:
                    # agent2向空格移动，agent1向agent2的原位置移动
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
            else:
                # agent1向空格移动（valid_state1 == 1）
                if valid_state2 == 0:
                    # agent2不能移动的情况，agent1正常移动
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
                elif valid_state2 == 1:
                    # agent2向空格移动，agent1正常移动
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
                else:
                    # agent2向agent1移动
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0

    def move_grasp(self, pos_agent1, pos_agent2, pos_goods, action_n):
        """
        当agent1与agent2共同抬着货物时，根据动作更新3者的位置
        :param pos_agent1:
        :param pos_agent2:
        :param pos_goods:
        :param action_n:
        :return:
        """
        if action_n[0] != action_n[1]:
            # 如果两个agent的动作不同，则各自单独移动，规则同move_not_grasp函数
            # self.move_not_grasp(pos_agent1, pos_agent2, action_n)

            # 如果两个agent的动作不同，则两个agent不动
            return
        else:
            # 两个agent的动作相同
            if action_n[0] == 0:
                # 两者都不动的情况
                return
            else:
                # 两者向同一方向移动

                new_pos_agent1 = (pos_agent1[0] + self.move_delta[action_n[0]][0],
                                  pos_agent1[1] + self.move_delta[action_n[0]][1])
                new_pos_agent2 = (pos_agent2[0] + self.move_delta[action_n[1]][0],
                                  pos_agent2[1] + self.move_delta[action_n[1]][1])
                new_pos_goods = (pos_goods[0] + self.move_delta[action_n[0]][0],
                                 pos_goods[1] + self.move_delta[action_n[0]][1])
                valid_state1 = self.valid(new_pos_agent1)
                valid_state2 = self.valid(new_pos_agent2)
                valid_goods = self.valid(new_pos_goods)

                if action_n[0] == 2 or action_n[0] == 4:
                    # 水平移动的情况
                    # 计算两个新位置，一个新位置应该在空格上，另一个应该在goods上，所以判断的方法为“至少有一个新位置在空格上即可”

                    if valid_state1 == 1 or valid_state2 == 1:
                        # 至少同一个新位置在空格上，则两个agent和goods正常移动
                        # 擦去旧位置
                        self.map[pos_agent1[0]][pos_agent1[1]] = 0
                        self.map[pos_agent2[0]][pos_agent2[1]] = 0
                        # 设置新位置
                        self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                        self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                        self.map[new_pos_goods[0]][new_pos_goods[1]] = 3
                    else:
                        # 两个新位置都没有空格，肯定不能移动
                        return
                else:
                    # 垂直移动
                    # 3个新位置都应该是空格
                    # 只要一个被挡住，整体就不能动
                    if valid_state1 == 1 and valid_state2 == 1 and valid_goods == 1:
                        # 擦去旧位置
                        self.map[pos_agent1[0]][pos_agent1[1]] = 0
                        self.map[pos_agent2[0]][pos_agent2[1]] = 0
                        self.map[pos_goods[0]][pos_goods[1]] = 0
                        # 设置新位置
                        self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                        self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                        self.map[new_pos_goods[0]][new_pos_goods[1]] = 3

    def valid(self, pos):
        """
        判断pos是否在地图中，并且是可移动到的位置（0或者4）
        :param pos:
        :return: 如果该位置为空，返回1；如果该位置被其他agent占据，返回2；移动出地图，或者移动向障碍或者goods，返回0
        """
        if pos[0] >= 0 and pos[0] < self.map_size[0] and pos[1] >= 0 and pos[1] < self.map_size[1]:
            if self.map[pos[0]][pos[1]] == 0:
                return 1
            if self.map[pos[0]][pos[1]] == 1 or self.map[pos[0]][pos[1]] == 2:
                return 2
        return 0


if __name__ == '__main__':
    cm = CMOTP()
    cm.reset()
    cm.render()
    move_str = ["still", 'up', 'right', 'down', 'left']
    import time
    import random
    while True:
        a = random.randint(0, 4)
        b = random.randint(0, 4)
        cm.step((a, b))
        print(move_str[a], move_str[b])
        cm.render()
        time.sleep(1)
    cm.close()
    # env = gym.make('CartPole-v0')
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break
    # env.close()
