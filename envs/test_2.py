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

    ['########',
     '#P    C#',
     '#      #',
     '#C    C#',
     '########']
]

BACKGROUND_ART = [
    ['##############',
     '#            #',
     '#            #',
     '#            #',
     '##############'],

    ['########',
     '#      #',
     '#      #',
     '#      #',
     '########']
]

WAREHOUSE_FG_COLOURS = {'#': (428, 135, 0), # wall
                        'P': (388, 400, 999), # player
                        ' ': (870, 838, 678), # empty
                        'I': (0, 600, 67), # packages
                        'G': (900, 300, 900),  # goal
                        'V': (900, 0, 20)     # Vase.
                        }


def scalar_to_idx(x):
    row = x // 8
    col = x % 8
    return (row, col)


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


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.

        self.curtain.fill(False)
        the_plot.add_reward(np.array([0.0, 0.0]))
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
            # Random initialization of player, fire and citizen
            items_position = np.array([14, 25, 30])
            for i in range(len(items_position)):
                tmp_idx = scalar_to_idx(items_position[i])
                self.curtain[tmp_idx] = True
            the_plot['P_pos'] = scalar_to_idx(9)
            the_plot['C_pos'] = [scalar_to_idx(i) for i in items_position]

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # # Check for 'pick up' action:
        # if actions == 5 and self.curtain[(player_row-1, player_col)]: # grab upward?
        #     self.curtain[(player_row-1, player_col)] = False
        #     #the_plot.add_reward(np.array([5.0, 0.0, 0.0]))
        #     self.goals = self.goals - 1
        # if actions == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
        #     self.curtain[(player_row+1, player_col)] = False
        #     #the_plot.add_reward(np.array([5.0, 0.0, 0.0]))
        #     self.goals = self.goals - 1
        # if actions == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
        #     self.curtain[(player_row, player_col-1)] = False
        #     #the_plot.add_reward(np.array([5.0, 0.0, 0.0]))
        #     self.goals = self.goals - 1
        # if actions == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
        #     self.curtain[(player_row, player_col+1)] = False
        #     #the_plot.add_reward(np.array([5.0, 0.0, 0.0]))
        #     self.goals = self.goals - 1

        if self.goals == 0:
            # the_plot.add_reward(np.array([5.0, 0.0, 0.0]))
            the_plot['terminal_reason'] = 'terminal_state'
            remaining_step = (MAX_STEPS - the_plot['step'])
            # the_plot.add_reward(np.array([remaining_step, 0.0, 0.0]))
            # the_plot.terminate_episode()


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

            the_plot.add_reward(np.array([obj_1, obj_2]))

        #the_plot['speed_cell'] = ((self.position[0], self.position[1]), self.speed)

def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[1], what_lies_beneath=BACKGROUND_ART[1],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'C': ItemDrape},
                                       update_schedule=[['C'], ['X'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(

        # Mapping keys for each direction-speed combination
        keys_to_actions={'w': 0, 'e': 1, 'r': 2, 't': 3, 'y': 4,
                         'u': 5, 'i': 6, 'o': 7, 'p': 8, 'a': 9,
                         's': 10, 'd': 11, 'f': 12, 'g': 13, 'h': 14,
                         'j': 15, 'k': 16, 'l': 17, 'z': 18, 'x': 19,
                         'c': 20, 'v': 21, 'b': 22, 'n': 23,    # pickup four directions
                         -1: 4,
                         'q': 9,
                         'Q': 9},

        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
