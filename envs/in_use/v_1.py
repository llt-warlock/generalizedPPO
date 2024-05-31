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

ITEMS = 3
DUST = 3
OBSTACLE = 3

region_0_0 = np.concatenate(
    [np.arange(13, 15), np.arange(23, 26), np.arange(34, 37)])
region_0_1 = np.concatenate(
    [np.arange(15, 18), np.arange(26, 29), np.arange(37, 40)])
region_0_2 = np.concatenate(
    [np.arange(18, 21), np.arange(29, 32), np.arange(40, 42)])


board_position = np.concatenate([np.arange(13, 23), np.arange(25, 35), np.arange(37, 47),
                                 np.arange(49, 59), np.arange(61, 71)])

board_position_below = np.concatenate([np.arange(73, 83), np.arange(85, 95), np.arange(97, 107),
                                       np.arange(109, 119), np.arange(121, 131)])

WAREHOUSE_ART = [
    ['###########',
     '#P        #',
     '#         #',
     '#        G#',
     '###########']
]

BACKGROUND_ART = [
    ['###########',
     '#         #',
     '#         #',
     '#         #',
     '###########']
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
    row = x // 11
    col = x % 11
    return (row, col)

def idx_to_scalar(row, col):
    x = row * 11 + col
    return x


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([-0.1, 0.0]))
        self._step_counter += 1
        the_plot['step'] = self._step_counter

        #print("step : ", self._step_counter)

        if the_plot.frame == 0:
            # Random initialization of player, fire and citizen

            random_r0 = np.random.choice(region_0_0, size=3, replace=False)
            random_r1 = np.random.choice(region_0_1, size=3, replace=False)
            random_r2 = np.random.choice(region_0_2, size=3, replace=False)

            item_positions = np.concatenate([random_r0, random_r1, random_r2])

            the_plot['C_pos'] = [scalar_to_idx(i) for i in item_positions]
            the_plot['P_pos'] = scalar_to_idx(12)

            the_plot['G_pos'] = scalar_to_idx(42)
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
            item_positions = the_plot['C_pos']
            for pos in item_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))


class DustDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(DustDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            switch_positions = the_plot['D_pos']
            for pos in switch_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))


class SwitchDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(SwitchDrape, self).__init__(curtain, character)

        self.number_of_switch = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            switch_positions = the_plot['switch_pos']
            for pos in switch_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)] and actions == 4:
            self.curtain[(player_row, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

            if idx_to_scalar(player_row, player_col) in board_position:
                the_plot['above_switch'] += 1
            elif idx_to_scalar(player_row, player_col) in board_position_below:
                the_plot['below_switch'] += 1


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        #the_plot.add_reward(np.array([0.0, 0.0]))

        if the_plot.frame == 0:
            goal_pos = the_plot['G_pos']
            self.curtain[goal_pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            self.curtain[(player_row, player_col)] = False
            # remaining_step = MAX_STEPS - the_plot['step']
            # #the_plot.add_reward(np.array([(1 + remaining_step * 0.1), 0.0, 0.0, 0.0]))
            # the_plot.add_reward(np.array([0.0, (remaining_step * 0.01), 0.0, 0.0, 0.0]))
            the_plot['terminal_reason'] = 'terminal_state'
            the_plot.terminate_episode()


class ObstacleDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ObstacleDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            obstacle_positions = the_plot['O_pos']
            for pos in obstacle_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # if self.curtain[(player_row, player_col)]:
        #     self.curtain[(player_row, player_col)] = False
        #     the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0]))


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
    # return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
    #                                    sprites={'P': PlayerSprite},
    #                                    drapes={'X': JudgeDrape, 'C': ItemDrape, 'V': DustDrape, 'F': SwitchDrape,
    #                                            'G': GoalDrape},
    #                                    update_schedule=[['C'], ['X'], ['V'], ['F'], ['G'], ['P']])

    # expert preferences
    return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'C': ItemDrape, 'G': GoalDrape
                                               },
                                       update_schedule=[['X'], ['C'], ['G'], ['P']])


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
