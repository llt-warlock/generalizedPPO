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
MAX_STEPS = 50

WAREHOUSE_ART = [
    ['##########',
     '#P       #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '##########']
]

BACKGROUND_ART = [
    ['##########',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '#        #',
     '##########']
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


BOARD = np.concatenate([np.arange(11, 19), np.arange(21, 29), np.arange(31, 39), np.arange(41, 49),
                       np.arange(51, 59), np.arange(61, 69), np.arange(71, 79), np.arange(81, 89)])


# BOARD = np.concatenate([np.arange(13, 23), np.arange(25, 35), np.arange(37, 47), np.arange(49, 59),
#                        np.arange(61, 71), np.arange(73, 83), np.arange(85, 95), np.arange(97, 107),
#                        np.arange(109, 119), np.arange(121, 131)])
V_5_TOP_REGION = np.concatenate(
    [np.arange(11,19), np.arange(21,29), np.arange(31,39)])
V_5_LEFT_REGION = np.concatenate(
    [np.arange(41,45), np.arange(51,55), np.arange(61,65), np.arange(71,75), np.arange(81,85)])
V_5_RIGHT_REGION = np.concatenate(
    [np.arange(45,49), np.arange(55,59), np.arange(65,69), np.arange(75,79), np.arange(85,89)])


def scalar_to_idx(x):
    row = x // 10
    col = x % 10
    return (row, col)


def idx_to_scalar(row, col):
    x = row * 10 + col
    return x

def v_5_idx_to_scalar(row, col):
    x = row * 10 + col
    return x

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
        the_plot['step'] = self._step_counter
        the_plot['left_item_A'] = 0
        the_plot['right_item_A'] = 0
        the_plot['top_item_A'] = 0
        the_plot['left_item_B'] = 0
        the_plot['right_item_B'] = 0
        the_plot['top_item_B'] = 0
        the_plot['left_item_C'] = 0
        the_plot['right_item_C'] = 0
        the_plot['top_item_C'] = 0

        if the_plot.frame == 0:
            if 'okla' in the_plot:
                item_A_positions = the_plot['item_A_pos']
                item_B_positions = the_plot['item_B_pos']
                item_C_positions = the_plot['item_C_pos']
                the_plot['P_pos'] = the_plot['P_pos']
            else:
                random_positions = np.random.choice(BOARD, size=19, replace=False)
                item_A_positions = random_positions[0:6]
                item_B_positions = random_positions[6:12]
                item_C_positions = random_positions[12:18]
                the_plot['P_pos'] = scalar_to_idx(random_positions[-1])

                the_plot['item_A_pos'] = [scalar_to_idx(i) for i in item_A_positions]
                the_plot['item_B_pos'] = [scalar_to_idx(i) for i in item_B_positions]
                the_plot['item_C_pos'] = [scalar_to_idx(i) for i in item_C_positions]

            the_plot['left_item_A'] = 0
            the_plot['right_item_A'] = 0
            the_plot['top_item_A'] = 0
            the_plot['left_item_B'] = 0
            the_plot['right_item_B'] = 0
            the_plot['top_item_B'] = 0
            the_plot['left_item_C'] = 0
            the_plot['right_item_C'] = 0
            the_plot['top_item_C'] = 0

            the_plot['item_A_left_max'] = 0
            the_plot['item_A_right_max'] = 0
            the_plot['item_A_top_max'] = 0
            the_plot['item_B_left_max'] = 0
            the_plot['item_B_right_max'] = 0
            the_plot['item_B_top_max'] = 0
            the_plot['item_C_left_max'] = 0
            the_plot['item_C_right_max'] = 0
            the_plot['item_C_top_max'] = 0

            for i in item_A_positions:
                if i in V_5_LEFT_REGION:
                    the_plot['item_A_left_max'] += 1
                elif i in V_5_RIGHT_REGION:
                    the_plot['item_A_right_max'] += 1
                elif i in V_5_TOP_REGION:
                    the_plot['item_A_top_max'] += 1

            for i in item_B_positions:
                if i in V_5_LEFT_REGION:
                    the_plot['item_B_left_max'] += 1
                elif i in V_5_RIGHT_REGION:
                    the_plot['item_B_right_max'] += 1
                elif i in V_5_TOP_REGION:
                    the_plot['item_B_top_max'] += 1

            for i in item_C_positions:
                if i in V_5_LEFT_REGION:
                    the_plot['item_C_left_max'] += 1
                elif i in V_5_RIGHT_REGION:
                    the_plot['item_C_right_max'] += 1
                elif i in V_5_TOP_REGION:
                    the_plot['item_C_top_max'] += 1



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

        the_plot['left_item_A'] = 0
        the_plot['right_item_A'] = 0
        the_plot['top_item_A'] = 0

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row - 1, player_col) in V_5_LEFT_REGION:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in V_5_RIGHT_REGION:
                the_plot['right_item_A'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in V_5_TOP_REGION:
                the_plot['top_item_A'] = 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row + 1, player_col) in V_5_LEFT_REGION:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in V_5_RIGHT_REGION:
                the_plot['right_item_A'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in V_5_TOP_REGION:
                the_plot['top_item_A'] = 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row, player_col - 1) in V_5_LEFT_REGION:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in V_5_RIGHT_REGION:
                the_plot['right_item_A'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in V_5_TOP_REGION:
                the_plot['top_item_A'] = 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row, player_col + 1) in V_5_LEFT_REGION:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in V_5_RIGHT_REGION:
                the_plot['right_item_A'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in V_5_TOP_REGION:
                the_plot['top_item_A'] = 1


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

        the_plot['left_item_B'] = 0
        the_plot['right_item_B'] = 0
        the_plot['top_item_B'] = 0

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0]))

            if idx_to_scalar(player_row - 1, player_col) in V_5_LEFT_REGION:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in V_5_RIGHT_REGION:
                the_plot['right_item_B'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in V_5_TOP_REGION:
                the_plot['top_item_B'] = 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0]))

            if idx_to_scalar(player_row + 1, player_col) in V_5_LEFT_REGION:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in V_5_RIGHT_REGION:
                the_plot['right_item_B'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in V_5_TOP_REGION:
                the_plot['top_item_B'] = 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0]))

            if idx_to_scalar(player_row, player_col - 1) in V_5_LEFT_REGION:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in V_5_RIGHT_REGION:
                the_plot['right_item_B'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in V_5_TOP_REGION:
                the_plot['top_item_B'] = 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0, 0.0]))

            if idx_to_scalar(player_row, player_col + 1) in V_5_LEFT_REGION:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in V_5_RIGHT_REGION:
                the_plot['right_item_B'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in V_5_TOP_REGION:
                the_plot['top_item_B'] = 1


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

        the_plot['left_item_C'] = 0
        the_plot['right_item_C'] = 0
        the_plot['top_item_C'] = 0

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0]))

            if idx_to_scalar(player_row - 1, player_col) in V_5_LEFT_REGION:
                the_plot['left_item_C'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in V_5_RIGHT_REGION:
                the_plot['right_item_C'] = 1
            elif idx_to_scalar(player_row - 1, player_col ) in V_5_TOP_REGION:
                the_plot['top_item_C'] = 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0]))

            if idx_to_scalar(player_row + 1, player_col) in V_5_LEFT_REGION:
                the_plot['left_item_C'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in V_5_RIGHT_REGION:
                the_plot['right_item_C'] = 1
            elif idx_to_scalar(player_row + 1, player_col ) in V_5_TOP_REGION:
                the_plot['top_item_C'] = 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0]))

            if idx_to_scalar(player_row, player_col - 1) in V_5_LEFT_REGION:
                the_plot['left_item_C'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in V_5_RIGHT_REGION:
                the_plot['right_item_C'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in V_5_TOP_REGION:
                the_plot['top_item_C'] = 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 1.0]))

            if idx_to_scalar(player_row, player_col + 1) in V_5_LEFT_REGION:
                the_plot['left_item_C'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in V_5_RIGHT_REGION:
                the_plot['right_item_C'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in V_5_TOP_REGION:
                the_plot['top_item_C'] = 1


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


def make_game(msg=None, seed=None, demo=False):
    # expert preferences
    # return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
    #                                    sprites={'P': PlayerSprite},
    #                                    drapes={'X': JudgeDrape, 'C': ItemADrape, 'V':ItemBDrape, 'G':ItemCDrape
    #                                            },
    #                                    update_schedule=[['X'], ['C'], ['V'], ['G'], ['P']])

    game = ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'C': ItemADrape, 'V':ItemBDrape, 'G':ItemCDrape},
                                       update_schedule=[['X'], ['C'], ['V'], ['G'], ['P']],
                                       )

    if msg is not None:
        game.the_plot['okla'] = "check"
        game.the_plot['item_A_pos'] = msg['item_A_pos']
        game.the_plot['item_B_pos'] = msg['item_B_pos']
        game.the_plot['item_C_pos'] = msg['item_C_pos']
        game.the_plot['P_pos'] = msg['P_pos']

    return game


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
                         #'p': 4,
                         'q': 9, 'Q': 9},

        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)