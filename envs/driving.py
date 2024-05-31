from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np
import pickle
import copy
import argparse
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites, drapes

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

X_LABEL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Y_LABEL = [1, 2, 3]

MAXTRIX_VALUE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

MAX_STEPS = 30

WAREHOUSE_ART = [
    ['##############',
     '#PCCC   C    #',  # 16, 17, 18, 22
     '# CCC C   CC #',  # column 12 (2)  , 30, 31 , 32, 34, 38, 39,
     '#     CCCCCCG#',  # 48, 49, 50, 51, 52, 53
     '##############'],

    ['##############',
     '#PC   C   C  #',
     '# C C C C C C#',
     '# C C C C C C#',
     '#   C   C   C#',
     '##############']
    ]

BACKGROUND_ART = [
    ['##############',
     '#            #',
     '#            #',
     '#            #',
     '##############'],

    ['##############',
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '##############']
    ]




WAREHOUSE_FG_COLOURS = {'#': (428, 135, 0),
                        'P': (388, 400, 999),
                        ' ': (870, 838, 678),
                        'C': (0, 600, 67),
                        'G': (900, 300, 900)
                        }

def scalar_to_idx(x):
    row = x // 14
    col = x % 14
    return (row, col)


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([0.0, 0.0, 0.0]))
        self._step_counter += 1

        if the_plot.frame == 0:
            the_plot['P_pos'] = (1, 1)
            the_plot['steps'] = 0
            goal_position = np.array([18, 22, 26, 43, 46, 54])
            for i in range(len(goal_position)):
                tmp_idx = scalar_to_idx(goal_position[i])
                self.curtain[tmp_idx] = True

            the_plot['G_pos'] = [scalar_to_idx(i) for i in goal_position]

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot.terminate_episode()

        the_plot['steps'] = self._step_counter


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

        self.goals = 6

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            goal_positions = the_plot['G_pos']
            for i in goal_positions:
                self.curtain[i] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # 添加奖赏
        if self.curtain[(player_row, player_col)]:
            self.curtain[(player_row, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.1]))
            self.goals -= 1

            if self.goals == 0:
                bonus = (MAX_STEPS - the_plot['steps'])
                the_plot.add_reward(np.array([0.0, 0.0, bonus]))
                the_plot.terminate_episode()


def get_reward(speed, type):
    slow_dict = {1: 1.0,
                 2: 0.6,
                 3: 0.2,
                 4: 0.1,
                 5: 0.0}
    fast_dict = {1: 0.0,
                 2: 0.1,
                 3: 0.2,
                 4: 0.6,
                 5: 1.0}
    if type == 'slow':
        return slow_dict[speed]
    elif type == 'fast':
        return fast_dict[speed]
    else:
        return 0


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')
        self.speed = 3
        # self.direction = 'RIGHTWARD'

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        if actions == 0:  # go upward?
            # if self.direction != 'UPWARD':
            #     the_plot.add_reward(np.array([-0.01, -0.01, 0.0]))
            self._north(board, the_plot)
            # self.direction = 'UPWARD'
            obj_1 = get_reward(self.speed, 'slow')
            obj_2 = get_reward(self.speed, 'fast')
            print("reward : ", obj_1, "  ", obj_2, " total reward: ", obj_1 + obj_2, "  speed : ", self.speed)
            the_plot.add_reward(np.array([obj_1, obj_2, 0.0]))
        elif actions == 1:  # go downward?
            # if self.direction != 'DOWNWARD':
            #     the_plot.add_reward(np.array([-0.01, -0.01, 0.0]))
            self._south(board, the_plot)
            # self.direction = 'DOWNWARD'
            obj_1 = get_reward(self.speed, 'slow')
            obj_2 = get_reward(self.speed, 'fast')
            print("reward : ", obj_1, "  ", obj_2, " total reward: ", obj_1 + obj_2, "  speed : ", self.speed)
            the_plot.add_reward(np.array([obj_1, obj_2, 0.0]))
        elif actions == 2:  # go leftward?
            # if self.direction != 'LEFTWARD':
            #     the_plot.add_reward(np.array([-0.01, -0.01, 0.0]))
            self._west(board, the_plot)
            # self.direction = 'LEFTWARD'
            obj_1 = get_reward(self.speed, 'slow')
            obj_2 = get_reward(self.speed, 'fast')
            print("reward : ", obj_1, "  ", obj_2, " total reward: ", obj_1 + obj_2, "  speed : ", self.speed)
            the_plot.add_reward(np.array([obj_1, obj_2, 0.0]))
        elif actions == 3:  # go rightward?
            # if self.direction != 'RIGHTWARD':
            #     the_plot.add_reward(np.array([-0.01, -0.01, 0.0]))
            # self.direction = 'RIGHTWARD'
            self._east(board, the_plot)
            obj_1 = get_reward(self.speed, 'slow')
            obj_2 = get_reward(self.speed, 'fast')
            print("reward : ", obj_1, "  ", obj_2, " total reward: ", obj_1 + obj_2, "  speed : ", self.speed)
            the_plot.add_reward(np.array([obj_1, obj_2, 0.0]))

        elif actions == 5 and self.speed > 1:  # speed down
            self.speed -= 1
            #the_plot.add_reward(np.array([0.01, -0.01, 0.0]))
        elif actions == 6 and self.speed < 5:  # speed up
            self.speed += 1
            #the_plot.add_reward(np.array([-0.01, 0.01, 0.0]))

        the_plot['speed_cell'] = ((self.position[0], self.position[1]), self.speed)


def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(art=LEVEL[0], what_lies_beneath=LEVEL[1], sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'G': GoalDrape},
                                       update_schedule=[['X'], ['G'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         's': 5,
                         'd': 6,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=COLOR)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
