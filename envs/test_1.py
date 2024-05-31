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
MAX_STEPS = 75

WAREHOUSE_ART = [
    ['##############',
     '#PCCC   C    #',  # 16, 17, 18, 22
     '# CCC C   CC #',  # column 12 (2)  , 30, 31 , 32, 34, 38, 39,
     '#     CCCCCCG#',  # 48, 49, 50, 51, 52, 53
     '##############'],

    ['##############',
     '#PCC        G#',
     '# CC       CC#',
     '#     CC   CC#',
     '#     CC     #',
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


def make_game(seed=None, demo=False):
    warehouse_art = WAREHOUSE_ART[1]
    what_lies_beneath = BACKGROUND_ART[1]
    sprites = {'P': PlayerSprite}

    if demo:
        raise NotImplementedError
    else:
        drapes = {'X': JudgeDrape}

    drapes['C'] = CitizenDrape
    drapes['G'] = GoalDrape

    update_schedule = [['C'],
                       ['G'],
                       ['X'],
                       ['P']]

    return ascii_art.ascii_art_to_game(
        warehouse_art, what_lies_beneath, sprites, drapes,
        update_schedule=update_schedule)


def scalar_to_idx(x):
    row = x // 14
    col = x % 14
    return (row, col)


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            goal_positions = the_plot['G_pos']

            self.curtain[goal_positions] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            self.curtain[player_row, player_col] = False
            the_plot.add_reward(np.array([0.0, 10.0, 0.0]))
            the_plot['terminal_reason'] = 'terminal_state'
            the_plot.terminate_episode()


class CitizenDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(CitizenDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            # Random initialization of player, fire and citizen
            items_position = np.array([16, 30, 44, 32, 46, 60, 20, 34, 48, 36, 50, 64, 24, 38, 52, 40, 54, 68])
            for i in range(len(items_position)):
                tmp_idx = scalar_to_idx(items_position[i])
                self.curtain[tmp_idx] = True
            the_plot['P_pos'] = scalar_to_idx(15)
            the_plot['G_pos'] = scalar_to_idx(26)
            the_plot['C_pos'] = [scalar_to_idx(i) for i in items_position]



        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:

            the_plot.add_reward(np.array([-1.0, 0.0, 0.0]))


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.


        print("actions : ", actions)

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


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([0.0, 0.0, -0.01]))
        # the_plot.add_reward(-0.1)
        self._step_counter += 1

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot['terminal_reason'] = 'max_step'
            the_plot.terminate_episode()


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(
        # keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
        #                  curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
        #                  -1: 4,
        #                  'q': 9, 'Q': 9},

        keys_to_actions={'w': 0, 'e': 1, 'r': 2, 't': 3, 'y': 4,
                         'u': 5, 'i': 6, 'o': 7, 'p': 8, 'a': 9,
                         's': 10, 'd': 11, 'f': 12, 'g': 13, 'h': 14,
                         'j': 15, 'k': 16, 'l': 17, 'z': 18, 'x': 19,
                         -1: 4,
                         'q': 9,
                         'Q': 9},

        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()
    if args.demo:
        main(demo=True)
    else:
        main(demo=False)
