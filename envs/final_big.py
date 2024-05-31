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

MAX_STEPS = 75

WAREHOUSE_ART = [
    ['######################',
     '#P                   #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '######################'],

    ['################################',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '################################']
]

BACKGROUND_ART = [
    ['######################',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '######################'],

    ['################################',
     '#                              #',
     '#                              #',
     '#                              #',  # 20 * 30
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '################################']
]

'''
objective 1: reaching specific positions in order
objective 2: avoiding random obstacles
objective 3: picking up packages
objective 4: moving fast
objective 5: moving slow 
'''

WAREHOUSE_FG_COLOURS = {'#': (428, 135, 0),
                        'P': (388, 400, 999),
                        ' ': (870, 838, 678),
                        'I': (0, 600, 67),
                        'G': (900, 300, 900),
                        'V': (900, 0, 20)  # Vase.
                        }


def scalar_to_idx(x):
    row = x // 32
    col = x % 32
    return (row, col)


def get_reward(speed, speed_type):
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

    if speed_type == 'slow':
        return slow_dict[speed]
    elif speed_type == 'fast':
        return fast_dict[speed]
    else:
        return 0


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.

        if the_plot.frame == 0:
            # Random initialization of player, fire and citizen
            vase_positions = np.array([91, 94, 73, 37, 57, 128, 360, 399, 403, 389, 392, 433])
            item_positions = np.array([69, 115, 157, 79, 84, 170, 289, 358, 377, 367, 371, 431])
            goal_positions = np.array([184, 218, 400, 460])
            the_plot['P_pos'] = scalar_to_idx(46)
            the_plot['V_pos'] = [scalar_to_idx(i) for i in vase_positions]
            the_plot['G_pos'] = [scalar_to_idx(i) for i in goal_positions]
            the_plot['I_pos'] = [scalar_to_idx(i) for i in item_positions]

        self.curtain.fill(False)
        the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        self._step_counter += 1
        the_plot['step'] = self._step_counter

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if self._step_counter == self._max_steps:
            the_plot['terminal_reason'] = 'max_step'
            the_plot.terminate_episode()

        the_plot['steps'] = self._step_counter


class ItemDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ItemDrape, self).__init__(curtain, character)

        self.goals = 3

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            item_positions = the_plot['I_pos']
            for i in range(len(item_positions)):
                tmp_idx = item_positions[i]
                self.curtain[tmp_idx] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # # Check for 'pick up' action:
        if actions == 20 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 5.0, 0.0, 0.0]))
        if actions == 21 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 5.0, 0.0, 0.0]))
        if actions == 22 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 5.0, 0.0, 0.0]))
        if actions == 23 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 5.0, 0.0, 0.0]))


class VaseDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(VaseDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            vase_positions = the_plot['V_pos']
            print("type : ", type(vase_positions), " ", vase_positions)
            for i in range(len(vase_positions)):
                tmp_idx = vase_positions[i]
                self.curtain[tmp_idx] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            the_plot.add_reward(np.array([0.0, -1.0, 0.0, 0.0, 0.0]))


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

        self.goal = 4

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            goal_positions = the_plot['G_pos']
            for i in range(len(goal_positions)):
                tmp_idx = goal_positions[i]
                self.curtain[tmp_idx] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            self.curtain[player_row, player_col] = False
            self.goal -= 1
            the_plot.add_reward(np.array([10.0, 0.0, 0.0, 0.0, 0.0]))
            if self.goal == 0:
                the_plot['terminal_reason'] = 'terminal_state'
                the_plot.terminate_episode()


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')
        self.direction_speed_map = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 1,
            6: 2,
            7: 3,
            8: 4,
            9: 5,
            10: 1,
            11: 2,
            12: 3,
            13: 4,
            14: 5,
            15: 1,
            16: 2,
            17: 3,
            18: 4,
            19: 5,
        }
        # self.direction = 'RIGHTWARD'

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        print("type : ", type(actions), "  ", actions)

        if [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19].__contains__(actions):

            if [0, 1, 2, 3, 4].__contains__(actions):
                # moving upward
                self._north(board, the_plot)
            elif [5, 6, 7, 8, 9].__contains__(actions):
                # moving downward
                self._south(board, the_plot)
            elif [10, 11, 12, 13, 14].__contains__(actions):
                # moving leftward
                self._west(board, the_plot)
            elif [15, 16, 17, 18, 19].__contains__(actions):
                # moving rightward
                self._east(board, the_plot)

            speed = self.direction_speed_map[actions]

            obj_1 = get_reward(speed, 'slow')
            obj_2 = get_reward(speed, 'fast')

            the_plot.add_reward(np.array([0.0, 0.0, 0.0, obj_1, obj_2]))


def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'I': ItemDrape, 'V': VaseDrape, 'G': GoalDrape},
                                       update_schedule=[['X'], ['V'], ['I'], ['G'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(

        # Mapping keys for each direction-speed combination
        keys_to_actions={'w': 0, 'e': 1, 'r': 2, 't': 3, 'y': 4,
                         'u': 5, 'i': 6, 'o': 7, 'p': 8, 'a': 9,
                         's': 10, 'd': 11, 'f': 12, 'g': 13, 'h': 14,
                         'j': 15, 'k': 16, 'l': 17, 'z': 18, 'x': 19,
                         'c': 20, 'v': 21, 'b': 22, 'n': 23,  # pickup four directions
                         -1: 4,
                         'q': 9,
                         'Q': 9},

        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
