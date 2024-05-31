from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np
import pickle
import copy
import argparse
import sys

import torch
from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# General Parameters
MAX_STEPS = 70

WAREHOUSE_ART = [
    ['############',
     '#P         #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '############']
]

BACKGROUND_ART = [
    ['############',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '#          #',
     '############']
]

WAREHOUSE_FG_COLOURS = {' ': (870, 838, 678),  # Floor.
                        '#': (428, 135, 0),  # Walls.
                        'C': (0, 600, 67),  # items.
                        'A': (850, 603, 270),  # region A.
                        'B': (850, 300, 300),  # expert B
                        'P': (388, 400, 999),  # The player.
                        'F': (500, 0, 500),  # switch.
                        'G': (999, 500, 0),  # destination.
                        'O': (0, 0, 0),  # OBSTACLE
                        'V': (900, 0, 20)}  # dust.


BOARD = np.concatenate([np.arange(13, 23), np.arange(25, 35), np.arange(37, 47), np.arange(49, 59),
                       np.arange(61, 71), np.arange(73, 83), np.arange(85, 95), np.arange(97, 107),
                        np.arange(109, 119), np.arange(121, 131)])



REGION_0 = np.concatenate(
    [np.arange(11, 15), np.arange(21, 25), np.arange(31, 35), np.arange(41, 45)])

REGION_1 = np.concatenate(
    [np.arange(15, 19), np.arange(25, 29), np.arange(35, 39), np.arange(45, 49)])

REGION_2 = np.concatenate(
    [np.arange(51, 55), np.arange(61, 65), np.arange(71, 75), np.arange(81, 85)])

REGION_3 = np.concatenate(
    [np.arange(55, 59), np.arange(65, 69), np.arange(75, 79), np.arange(85, 89)])


def scalar_to_idx(x):
    row = x // 12
    col = x % 12
    return (row, col)


def idx_to_scalar(row, col):
    x = row * 12 + col
    return x


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0]))
        self._step_counter += 1
        the_plot['step'] = self._step_counter

        if the_plot.frame == 0:
            # small
            random_positions = np.random.choice(BOARD, size=25, replace=False)
            item_A_positions = random_positions[0:6]
            item_B_positions = random_positions[6:12]
            item_C_positions = random_positions[12:18]
            item_D_positions = random_positions[18:24]

            the_plot['P_pos'] = scalar_to_idx(random_positions[-1])

            the_plot['item_A_pos'] = [scalar_to_idx(i) for i in item_A_positions]
            the_plot['item_B_pos'] = [scalar_to_idx(i) for i in item_B_positions]
            the_plot['item_C_pos'] = [scalar_to_idx(i) for i in item_C_positions]
            the_plot['item_D_pos'] = [scalar_to_idx(i) for i in item_D_positions]

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot['terminal_reason'] = 'max_step'
            the_plot.terminate_episode()


class ItemADrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ItemADrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['item_A_pos']
            for pos in positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0, 0.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0, 0.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0, 0.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0, 0.0]))


class ItemBDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ItemBDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['item_B_pos']
            for pos in positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0, 0.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0, 0.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0, 0.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0, 0.0]))


class ItemCDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ItemCDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['item_C_pos']
            for pos in positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col


        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0, 0.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0, 0.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0, 0.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0, 0.0]))



class ItemDDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ItemDDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['item_D_pos']
            for pos in positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col


        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 1.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 1.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 1.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 1.0]))


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

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
    # expert preferences
    return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'C': ItemADrape, 'V':ItemBDrape, 'G':ItemCDrape, 'F':ItemDDrape
                                               },
                                       update_schedule=[['X'], ['C'], ['V'], ['G'], ['F'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(

        # Mapping keys for each direction-speed combination
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         'w': 5,
                         's': 6,
                         'a': 7,
                         'd': 8,
                         -1: 4,
                         'q': 9, 'Q': 9},

        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)