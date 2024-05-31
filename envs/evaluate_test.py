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
from pycolab.prefab_parts import sprites as prefab_sprites

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# General Parameters
MAX_STEPS = 20


WAREHOUSE_ART = [
    ['#####', # 0 - 4
     '#P  #', # 5 - 9
     '#   #', # 10 - 14  # primary task  打卡机
     '#  G#', # 15 - 19
     '#####'],
    ]


BACKGROUND_ART = [
    ['#####',
     '#   #',
     '#   #',  # primary task  打卡机
     '#   #',
     '#####'],
]

WAREHOUSE_FG_COLOURS = {' ': (870, 838, 678),  # Floor.
                        '#': (428, 135, 0),  # Walls.
                        'C': (0, 600, 67),  # Citizen.
                        'x': (850, 603, 270),  # Unused.
                        'P': (388, 400, 999),  # The player.
                        'F': (500, 0, 500),  # Mail.
                        'G': (999, 500, 0),  # Street.
                        'V': (900, 0, 20)}  # Vase.


def scalar_to_idx(x):
    row = x // 5
    col = x % 5
    return (row, col)


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            the_plot['P_pos'] = scalar_to_idx(6)
            goal_pos = scalar_to_idx(18)
            self.curtain[goal_pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            self.curtain[(player_row, player_col)] = False
            remaining_step = MAX_STEPS - the_plot['step']
            the_plot.add_reward(np.array([(1 + remaining_step*0.1)]))
            the_plot['terminal_reason'] = 'terminal_state'
            the_plot.terminate_episode()


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([0.]))
        self._step_counter += 1
        the_plot['step'] = self._step_counter

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot['terminal_reason'] = 'max_step'
            the_plot.terminate_episode()


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#H.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        if actions == 0:  # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)


def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape,
                                               'G': GoalDrape},
                                       update_schedule=[ ['X'], ['G'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(

        # # Mapping keys for each direction-speed combination
        # keys_to_actions={'w': 0, 'e': 1, 'r': 2, 't': 3, 'y': 4,
        #                  'u': 5, 'i': 6, 'o': 7, 'p': 8, 'a': 9,
        #                  's': 10, 'd': 11, 'f': 12, 'g': 13, 'h': 14,
        #                  'j': 15, 'k': 16, 'l': 17, 'z': 18, 'x': 19,
        #                  'c': 20, 'v': 21, 'b': 22, 'n': 23,    # pickup four directions
        #                  -1: 4,
        #                  'q': 9,
        #                  'Q': 9},

        # Mapping keys for each direction-speed combination
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         -1: 4,
                         'q': 9, 'Q': 9},

        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
