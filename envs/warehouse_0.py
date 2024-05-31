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

# region_A_info = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
#                          (2, 1), (2, 2), (2, 3),
#                          (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
#                          (4, 1), (4, 2), (4, 3),
#                          (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10)]
#
# region_B_info = [(1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20),
#                  (2, 18), (2, 19), (2, 20),
#                  (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20),
#                  (4, 18), (4, 19), (4, 20),
#                  (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20)]


MAX_STEPS = 190
LEVEL = [

    ['######################',  # row 6
     '#PAAAAAAAAAAAAAAAAAAA#',
     '#$AAAAAAAAAAAAAAAAAAA#',
     '#&AAAAAAAAAAAAAAAAAAA#',
     '#AAAAAAAAAAAAAAAAAAAA#',
     '#AAAAAAAAAAAAAAAAAAAA#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '#BBBBBBBBBBCCCCCCCCCC#',
     '######################'],

    ['######################',
     '#A                   #',
     '#A                   #',
     '#A                   #',
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

    ['######################',
     '#PAAAAAAAAABBBBBBBBBB#',
     '#AAA##############BBB#',   # col 22
     '#AAAAAAAAAABBBBBBBBBB#',   # row 7
     '#AAA##############BBB#',
     '#AAAAAAAAAABBBBBBBBBB#',
     '######################'],

    ['######################',
     '#A                   #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '######################'],

    ['################################',
     '#PAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',   # 3 * 30
     '#AAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '#AAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '################################'],

    ['################################',
     '#A                             #',
     '#                              #',
     '#                              #',
     '################################']

]

COLOR = {'#': (428, 135, 0),
         'P': (388, 400, 999),
         '$': (870, 838, 678),
         '&': (900, 0, 20),
         'A': (0, 600, 67),
         'B': (900, 300, 900)
         }

class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([0., 0.]))
        self._step_counter += 1

        if the_plot.frame == 0:
            the_plot['P_pos'] = (1, 1)
            the_plot['ap_pos'] = []
            the_plot['bp_pos'] = []

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot.terminate_episode()

class Package_A_Drape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(Package_A_Drape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if actions == 5 and not self.curtain[(player_row, player_col)] and not (player_row, player_col) in the_plot['bp_pos']:
            self.curtain[(player_row, player_col)] = True
            the_plot['ap_pos'].append((player_row, player_col))
            the_plot.add_reward(np.array([1.0, 0.0]))




class Package_B_Drape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(Package_B_Drape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if actions == 6 and not self.curtain[(player_row, player_col)] and not (player_row, player_col) in the_plot['ap_pos']:
            self.curtain[(player_row, player_col)] = True
            the_plot['bp_pos'].append((player_row, player_col))
            the_plot.add_reward(np.array([0.0, 1.0]))


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

        self.direction = None

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        if actions == 0:  # go upward?
            if self.direction != 'UPWARD':
                the_plot.add_reward(np.array([-0.1, -0.1]))
            self.direction = 'UPWARD'
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            if self.direction != 'DOWNWARD':
                the_plot.add_reward(np.array([-0.1, -0.1]))
            self.direction = 'DOWNWARD'
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            if self.direction != 'LEFTWARD':
                the_plot.add_reward(np.array([-0.1, -0.1]))
            self.direction = 'LEFTWARD'
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            if self.direction != 'RIGHTWARD':
                the_plot.add_reward(np.array([-0.1, -0.1]))
            self.direction = 'RIGHTWARD'
            self._east(board, the_plot)


def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(art=LEVEL[4], what_lies_beneath=LEVEL[5], sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, '$': Package_A_Drape, '&': Package_B_Drape},
                                       update_schedule=[['X'], ['$'], ['&'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         's': 5,
                         'd': 6,
                         -1: 4,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=COLOR)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
