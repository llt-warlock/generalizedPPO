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

PRIMARY = 6
ITEM_A = 6
ITEM_B = 6

region_left = np.concatenate(
    [np.arange(11, 15), np.arange(21, 25), np.arange(31, 35), np.arange(41, 45), np.arange(51, 55),
     np.arange(61, 65), np.arange(71, 75), np.arange(81, 85)])

region_left_p = np.concatenate(
    [np.arange(12, 15), np.arange(21, 25), np.arange(31, 35), np.arange(41, 45), np.arange(51, 55),
     np.arange(61, 65), np.arange(71, 75), np.arange(81, 85)])

region_right = np.concatenate(
    [np.arange(15, 19), np.arange(25, 29), np.arange(35, 39), np.arange(45, 49), np.arange(55, 59),
     np.arange(65, 69), np.arange(75, 79), np.arange(85, 89)])


board_position =  np.concatenate(
    [np.arange(11, 19), np.arange(21, 29), np.arange(31, 39), np.arange(41, 49), np.arange(51, 59),
     np.arange(61, 69), np.arange(71, 79), np.arange(81, 89)])

WAREHOUSE_ART = [
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


def scalar_to_idx(x):
    row = x%8
    col = int(np.floor(x/8))
    return (row+1, col+1)

def scalar_to_idx_0(x):
    row = x // 10
    col = x % 10
    return (row, col)


def idx_to_scalar(row, col):
    x = row * 10 + col
    return x


import matplotlib.pyplot as plt





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

        #print("step : ", self._step_counter)
        the_plot['left_item_A'] = 0
        the_plot['left_item_B'] = 0
        the_plot['right_item_A'] = 0
        the_plot['right_item_B'] = 0

        if the_plot.frame == 0:
            the_plot['A_left_total'] = 0
            the_plot['A_right_total'] = 0
            the_plot['B_left_total'] = 0
            the_plot['B_right_total'] = 0
            the_plot['left_item_A'] = 0
            the_plot['right_item_A'] = 0
            the_plot['left_item_B'] = 0
            the_plot['right_item_B'] = 0

            # # version 1
            # # Random initialization of player, fire and citizen
            # random_positions = np.random.choice(8 * 8, size=PRIMARY + ITEM_A + ITEM_B + 1, replace=False)
            # the_plot['G_pos'] = [scalar_to_idx(i) for i in random_positions[0:PRIMARY]]
            # the_plot['C_pos'] = [scalar_to_idx(i) for i in random_positions[PRIMARY:PRIMARY+ITEM_A]]
            # the_plot['D_pos'] = [scalar_to_idx(i) for i in random_positions[PRIMARY+ITEM_A:PRIMARY+ITEM_A+ITEM_B]]
            # the_plot['P_pos'] = scalar_to_idx(random_positions[-1])
            #
            # for i in the_plot['C_pos']:
            #     scalar = idx_to_scalar(i[0], i[1])
            #     if scalar in region_left:
            #         the_plot['A_left_total'] += 1
            #     elif scalar in region_right:
            #         the_plot['A_right_total'] += 1
            #
            # for i in the_plot['D_pos']:
            #     scalar = idx_to_scalar(i[0], i[1])
            #     if scalar in region_left:
            #         the_plot['B_left_total'] += 1
            #     elif scalar in region_right:
            #         the_plot['B_right_total'] += 1
            # print("#########")
            # print("total left A : ", the_plot['A_left_total'])
            # print("total right A : ", the_plot['A_right_total'])
            # print("total left B : ", the_plot['B_left_total'])
            # print("total right B : ", the_plot['B_right_total'])
            # print("#########")

            random_positions_left = np.random.choice(region_left, size=12, replace=False)
            random_positions_right = np.random.choice(region_right, size=12, replace=False)
            item_positions = np.concatenate([random_positions_left[0:6], random_positions_right[0:6]])
            dust_positions = np.concatenate([random_positions_left[6:12], random_positions_right[6:12]])

            # Find elements that are common in a1 and either a2 or a3
            common_elements = np.union1d(random_positions_left, random_positions_right)

            # Remove common elements from a1
            rest_positions = np.setdiff1d(board_position, common_elements)
            player_position = np.random.choice(rest_positions, size=1, replace=False)

            the_plot['C_pos'] = [scalar_to_idx_0(i) for i in item_positions]
            the_plot['D_pos'] = [scalar_to_idx_0(i) for i in dust_positions]
            #("player positions : ", player_position)
            the_plot['P_pos'] = [scalar_to_idx_0(i) for i in player_position][-1]
        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot['terminal_reason'] = 'max_step'
            the_plot.terminate_episode()


class ItemDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ItemDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['C_pos']
            for pos in positions:
                self.curtain[pos] = True

        the_plot['left_item_A'] = 0
        the_plot['right_item_A'] = 0

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))

            if idx_to_scalar(player_row - 1, player_col) in region_left:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in region_right:
                the_plot['right_item_A'] = 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))

            if idx_to_scalar(player_row + 1, player_col) in region_left:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in region_right:
                the_plot['right_item_A'] = 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))

            if idx_to_scalar(player_row, player_col - 1) in region_left:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in region_right:
                the_plot['right_item_A'] = 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))

            if idx_to_scalar(player_row, player_col + 1) in region_left:
                the_plot['left_item_A'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in region_right:
                the_plot['right_item_A'] = 1

class DustDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(DustDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['D_pos']
            for pos in positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col
        the_plot['left_item_B'] = 0
        the_plot['right_item_B'] = 0

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

            if idx_to_scalar(player_row - 1, player_col) in region_left:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row - 1, player_col) in region_right:
                the_plot['right_item_B'] = 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

            if idx_to_scalar(player_row + 1, player_col) in region_left:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row + 1, player_col) in region_right:
                the_plot['right_item_B'] = 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

            if idx_to_scalar(player_row, player_col - 1) in region_left:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row, player_col - 1) in region_right:
                the_plot['right_item_B'] = 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

            if idx_to_scalar(player_row, player_col + 1) in region_left:
                the_plot['left_item_B'] = 1
            elif idx_to_scalar(player_row, player_col + 1) in region_right:
                the_plot['right_item_B'] = 1


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            positions = the_plot['G_pos']
            for pos in positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row - 1, player_col) in region_left:
                the_plot['left_item_P'] = 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row + 1, player_col) in region_left:
                the_plot['left_item_P'] = 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row, player_col - 1) in region_left:
                the_plot['left_item_P'] = 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

            if idx_to_scalar(player_row, player_col + 1) in region_left:
                the_plot['left_item_P'] = 1



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
                                       drapes={'X': JudgeDrape, 'C': ItemDrape, 'V':DustDrape
                                               },
                                       update_schedule=[['X'], ['C'], ['V'], ['P']])


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
