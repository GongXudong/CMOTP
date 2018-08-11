import unittest
from env.cmotp import CMOTP
import gym


class TestCMOTP(unittest.TestCase):

    def test_get_entity_pos(self):
        cm = CMOTP()
        cm.reset()
        self.assertEqual(cm.get_entity_pos(), ((5, 0), (5, 4), (2, 3)))

    def test_grasp_goods(self):
        cm = CMOTP()
        # test Horizontal
        self.assertTrue(cm.grasp_goods((1, 1), (1, 3), (1, 2)))
        self.assertFalse(cm.grasp_goods((1, 0), (1, 4), (1, 2)))
        self.assertFalse(cm.grasp_goods((1, 0), (1, 5), (1, 2)))

        # test Vertical
        self.assertTrue(cm.grasp_goods((0, 2), (2, 2), (1, 2)))
        self.assertFalse(cm.grasp_goods((1, 3), (4, 3), (3, 3)))
        self.assertFalse(cm.grasp_goods((1, 3), (5, 3), (3, 3)))

        # test general
        self.assertFalse(cm.grasp_goods((0, 0), (2, 2), (1, 1)))
        self.assertFalse(cm.grasp_goods((3, 1), (3, 2), (3, 3)))

    def test_move(self):
        cm = CMOTP()
        cm.reset()
        cm.move([2, 4])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 1, 0, 2, 0, 0, -1]])
        cm.move([4, 2])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [1, 0, 0, 0, 2, 0, -1]])
        cm.move([2, 4])
        cm.move([1, 4])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 1, 2, 0, 0, 0, -1]])
        cm.move([2, 0])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 1, 2, 0, 0, 0, -1]])
        # 跟随运动的情况
        cm.move([2, 1])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 2, -1, -1, -1, -1],
                                  [0, 0, 1, 0, 0, 0, -1]])
        cm.move([0, 1])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 2, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 1, 0, 0, 0, -1]])
        # 移动向同一个格子的情况
        cm.move([1, 3])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 2, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 1, 0, 0, 0, -1]])
        # 携带货物的情况
        cm.map = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 1, 3, 2, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, 0, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0, -1]]
        # 携带货物向右移动，右侧为空格
        cm.move([2, 2])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 1, 3, 2, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        # 携带货物，向上移动，agent1上面有障碍
        cm.move([1, 1])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 2, 0],
                                  [0, 0, 0, 1, 3, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        cm.move([0, 3])
        # 携带货物向右移动，右侧为空格
        cm.move([2, 2])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 3, 2],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        # 携带货物向右移动，右侧出地图
        cm.move([2, 2])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 3, 2],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        # 携带货物，向上移动
        cm.move([1, 1])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 1, 3, 2],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        cm.move([1, 1])
        cm.move([2, 2])
        cm.move([1, 1])
        cm.move([4, 4])
        self.assertEqual(cm.map, [[0, 0, 0, 1, 3, 2, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        cm.move([4, 4])
        cm.move([3, 3])
        self.assertEqual(cm.map, [[0, 0, 0, 3, 0, 0, 0],
                                  [0, 0, 1, -1, 2, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])
        # 携带货物，但是运动方向不同
        cm.map = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 1, 3, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, 0, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0, -1]]
        cm.move([1, 2])
        self.assertEqual(cm.map, [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 1, 3, 0, 2],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [-1, -1, 0, -1, -1, -1, -1],
                                  [0, 0, 0, 0, 0, 0, -1]])

    def test_step(self):
        cm = CMOTP()
        cm.reset()
        cm.map = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 3, 0, 0, 0],
                  [0, 0, 1, 0, 2, 0, 0],
                  [-1, -1, 0, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0, -1]]
        self.assertEqual(cm.step([1, 1]), (((2, 2, cm.GRASP_STATE.GRASPING_LEFT.value),
                                            (2, 4, cm.GRASP_STATE.GRASPING_RIGHT.value)),
                                           (1., 1.), (False, False), ({}, {})))
        cm.step([2, 2])
        cm.step([2, 2])
        cm.step([1, 1])
        cm.step([1, 1])
        self.assertEqual(cm.step([4, 4]), (((0, 3, cm.GRASP_STATE.GRASPING_LEFT.value),
                                            (0, 5, cm.GRASP_STATE.GRASPING_RIGHT.value)),
                                           (10., 10.), (True, True), ({}, {})))

        cm.map = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 1, 3, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, 0, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0, -1]]
        self.assertEqual(cm.step([1, 4]), (((2, 3, cm.GRASP_STATE.GRASPING_LEFT.value),
                                            (2, 5, cm.GRASP_STATE.GRASPING_RIGHT.value)),
                                           (1., 1.), (False, False), ({}, {})))

        cm.map = [[0, 0, 0, 0, 0, 2, 0],
                  [0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [-1, -1, 0, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0, -1]]

        self.assertEqual(cm.step([2, 3]), (((3, 5, cm.GRASP_STATE.FREE.value),
                                            (1, 5, cm.GRASP_STATE.FREE.value)),
                                           (1., 1.), (False, False), ({}, {})))

    def test(self):
        action_spaces = gym.spaces.Discrete(2)
        self.assertTrue(action_spaces.contains(0) and action_spaces.contains(1) and not action_spaces.contains(2))


if __name__ == '__main__':

    unittest.main()


