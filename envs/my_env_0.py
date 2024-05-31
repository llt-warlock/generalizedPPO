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

LEVEL = [
    ['##################################################################################',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#',
     '##################################################################################'],

    ['##################################################################################',
     '#                                                                                #',
     '#        P                                                                       #',
     '#                                                                                #',
     '#                                                                                #',
     '#                                                                                #',
     '#                                                                                #',
     '#                                                                                #',
     '#                                                                                #',
     '#                                                                                #',
     '#                                                                                #',
     '##################################################################################']
]

COLOR = {'#': (428, 135, 0),
         '$': (870, 838, 678),
         'P': (388, 400, 999),
         '&': (1, 1, 1)
         }


def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(
        art=LEVEL[1], what_lies_beneath=LEVEL[0], sprites={'P': PlayerSprite},
        update_schedule=[['P']]
    )


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        # if the_plot.frame == 0:
        #     self._teleport(the_plot['P_pos'])

        if actions == 0:  # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         'w': 5,
                         's': 6,
                         'a': 7,
                         'd': 8,
                         -1: 4,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=COLOR)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
