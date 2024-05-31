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


region_0 = np.concatenate(
    [np.arange(28, 30), np.arange(53, 56), np.arange(79, 82), np.arange(105, 108)])
region_1 = np.concatenate(
    [np.arange(30, 33), np.arange(56, 59), np.arange(82, 85), np.arange(108, 111)])
region_2 = np.concatenate(
    [np.arange(33, 36), np.arange(59, 62), np.arange(85, 88), np.arange(111, 114)])
region_3 = np.concatenate(
    [np.arange(36, 39), np.arange(62, 65), np.arange(88, 91), np.arange(114, 117)])
region_4 = np.concatenate(
    [np.arange(39, 42), np.arange(65, 68), np.arange(91, 94), np.arange(117, 120)])
region_5 = np.concatenate(
    [np.arange(42, 45), np.arange(68, 71), np.arange(94, 97), np.arange(120, 123)])
region_6 = np.concatenate(
    [np.arange(45, 48), np.arange(71, 74), np.arange(97, 100), np.arange(123, 126)])
region_7 = np.concatenate(
    [np.arange(48, 51), np.arange(74, 77), np.arange(100, 103), np.arange(126, 128)])


WAREHOUSE_ART = [
    ['##########################',
     '#P                       #',
     '#                        #',  # primary task  打卡机
     '#                        #',
     '#                       G#',
     '##########################'],



    ['############',
     '#          #',
     '# C C  C C #',
     '#          #',  # expert A collecting items at low speed
     '#  C    C  #',
     '#          #',
     '#          #',
     '#  C    C  #',
     '#          #',
     '# C C  C C #',
     '#         G#',
     '############'],

    ['############',
     '#          #',
     '# C C  C C #',
     '#  G    G  #',  # expert B high speed and avoiding obstacles
     '#  C    C  #',
     '#          #',
     '#          #',
     '#  C    C  #',
     '#  G    G  #',
     '# C C  C C #',
     '#         G#',
     '############'],
]

BACKGROUND_ART = [

    ['##########################',
     '#                        #',
     '#                        #',  # primary task  打卡机
     '#                        #',
     '#                        #',
     '##########################']
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
    row = x // 26
    col = x % 26
    return (row, col)


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        # primary, speed fast, speed slow, item A, item B
        the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        self._step_counter += 1
        the_plot['step'] = self._step_counter

        if the_plot.frame == 0:
            # Random initialization of player, fire and citizen

            random_r0 = np.random.choice(region_0, size=5, replace=False)
            random_r1 = np.random.choice(region_1, size=5, replace=False)
            random_r2 = np.random.choice(region_2, size=5, replace=False)
            random_r3 = np.random.choice(region_3, size=5, replace=False)
            random_r4 = np.random.choice(region_4, size=5, replace=False)
            random_r5 = np.random.choice(region_5, size=5, replace=False)
            random_r6 = np.random.choice(region_6, size=5, replace=False)
            random_r7 = np.random.choice(region_7, size=5, replace=False)

            switch_positions = np.concatenate([random_r0[0:1], random_r1[0:1], random_r2[0:1], random_r3[0:1], random_r4[0:1], random_r5[0:1], random_r6[0:1], random_r7[0:1]])
            item_positions = np.concatenate([random_r0[1:3], random_r1[1:3], random_r2[1:3], random_r3[1:3], random_r4[1:3], random_r5[1:3], random_r6[1:3], random_r7[1:3]])
            dust_positions = np.concatenate([random_r0[3:5], random_r1[3:5], random_r2[3:5], random_r3[3:5], random_r4[3:5], random_r5[3:5], random_r6[3:5], random_r7[3:5]])

            the_plot['switch_pos'] = [scalar_to_idx(i) for i in switch_positions]
            the_plot['C_pos'] = [scalar_to_idx(i) for i in item_positions]
            the_plot['P_pos'] = scalar_to_idx(27)
            the_plot['G_pos'] = scalar_to_idx(128)
            the_plot['D_pos'] = [scalar_to_idx(i) for i in dust_positions]

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
        if actions == 12 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 2.0, 0.0]))

        if actions == 13 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 2.0, 0.0]))

        if actions == 14 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 2.0, 0.0]))

        if actions == 15 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 2.0, 0.0]))


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
        if actions == 12 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0, 2.0]))

        if actions == 13 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0, 2.0]))

        if actions == 14 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0, 2.0]))

        if actions == 15 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False
            the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.0, 2.0]))


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

        if self.curtain[(player_row, player_col)] and actions == 16:
            self.curtain[(player_row, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        if the_plot.frame == 0:
            goal_pos = the_plot['G_pos']
            self.curtain[goal_pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            self.curtain[(player_row, player_col)] = False
            remaining_step = MAX_STEPS - the_plot['step']
            the_plot.add_reward(np.array([(1 + remaining_step*0.1), 0.0, 0.0, 0.0, 0.0]))
            the_plot['terminal_reason'] = 'terminal_state'
            the_plot.terminate_episode()


def get_reward(speed, type):
    fast_dict = {0: 1.0,
                 3: 1.0,
                 6: 1.0,
                 9: 1.0,
                 1: 0.0,
                 4: 0.0,
                 7: 0.0,
                 10: 0.0,
                 2: -1.0,
                 5: -1.0,
                 8: -1.0,
                 11: -1.0}
    slow_dict = {0: -1.0,
                 3: -1.0,
                 6: -1.0,
                 9: -1.0,
                 1: 0.0,
                 4: 0.0,
                 7: 0.0,
                 10: 0.0,
                 2: 1.0,
                 5: 1.0,
                 8: 1.0,
                 11: 1.0}

    if type == 'slow':
        return slow_dict[speed]
    elif type == 'fast':
        return fast_dict[speed]
    else:
        return 0


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#H.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        if actions in [0,1,2]:  # go upward?
            self._north(board, the_plot)
            the_plot.add_reward(np.array([0.0, get_reward(actions, 'fast'), get_reward(actions, 'slow'), 0.0, 0.0]))
        elif actions in [3,4,5]:  # go downward?
            self._south(board, the_plot)
            the_plot.add_reward(np.array([0.0, get_reward(actions, 'fast'), get_reward(actions, 'slow'), 0.0, 0.0]))
        elif actions in [6,7,8]:  # go leftward?
            self._west(board, the_plot)
            the_plot.add_reward(np.array([0.0, get_reward(actions, 'fast'), get_reward(actions, 'slow'), 0.0, 0.0]))
        elif actions in [9,10,11]:  # go rightward?
            self._east(board, the_plot)
            the_plot.add_reward(np.array([0.0, get_reward(actions, 'fast'), get_reward(actions, 'slow'), 0.0, 0.0]))


def make_game(seed=None, demo=False):

    # expert preferences
    return ascii_art.ascii_art_to_game(art=WAREHOUSE_ART[0], what_lies_beneath=BACKGROUND_ART[0],
                                       sprites={'P': PlayerSprite},
                                       drapes={'X': JudgeDrape, 'C': ItemDrape, 'V': DustDrape, 'F':SwitchDrape, 'G':GoalDrape
                                               },
                                       update_schedule=[['X'], ['F'], ['G'], ['C'], ['V'], ['P']])


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(

        # Mapping keys for each direction-speed combination
        keys_to_actions={'w': 0, 'e': 1, 'r': 2,  # upward with 3 speed
                         't': 3, 'y': 4, 'u': 5,  # downward with 3 speed
                         'i': 6, 'o': 7, 'p': 8,  # leftward
                         'a': 9, 's': 10, 'd': 11,   # rightward with 5 speed
                         'f': 12, 'g': 13, 'h': 14, 'j': 15,    # pickup four directions
                         'k': 16,  # pressing
                         'q': 9,
                         'Q': 9},


        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
