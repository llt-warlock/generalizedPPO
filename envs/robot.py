from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import math

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

    ['##############',
     '#P CC CC CC C#',
     '# C        C #',  # column 12 (2)
     '#C CC CC CC G#',
     '##############'],

    ['##############',
     '#            #',
     '#            #',
     '#            #',
     '##############'],

    ['######################',
     '#P                   #',
     '#                    #',
     '#                    #',
     '#                   G#',
     '######################'],

    ['######################',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '######################']

]

COLOR = {'#': (428, 135, 0),
         'P': (388, 400, 999),
         ' ': (870, 838, 678),
         'C': (0, 600, 67),
         'G': (900, 300, 900)
         }


def scalar_to_idx(x):
    row = x // 22
    col = x % 22
    return (row, col)


array_0 = [scalar_to_idx(i) for i in
           np.concatenate([np.arange(23, 28), np.arange(45, 50), np.arange(67, 72), np.arange(89, 94)])]
array_1 = [scalar_to_idx(i) for i in
           np.concatenate([np.arange(28, 33), np.arange(50, 55), np.arange(72, 77), np.arange(94, 99)])]
array_2 = [scalar_to_idx(i) for i in
           np.concatenate([np.arange(33, 38), np.arange(55, 60), np.arange(77, 82), np.arange(99, 104)])]
array_3 = [scalar_to_idx(i) for i in
           np.concatenate([np.arange(38, 43), np.arange(60, 65), np.arange(82, 87), np.arange(104, 109)])]

MAX_STEPS = 112


def distance(current_position, goal_position):
    current_x = current_position[0]
    current_y = current_position[1]
    goal_x = goal_position[0]
    goal_y = goal_position[1]
    return math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)

        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        #the_plot.add_reward(np.array([0.0, 0.0, -0.01]))
        the_plot.add_reward(np.array([0.0]))
        self._step_counter += 1
        the_plot['steps'] = self._step_counter

        if the_plot.frame == 0:
            the_plot['P_pos'] = (1, 1)

            array_area_0 = np.concatenate([np.arange(24, 28), np.arange(45, 50), np.arange(67, 72), np.arange(89, 94)])
            array_area_1 = np.concatenate([np.arange(28, 33), np.arange(50, 55), np.arange(72, 77), np.arange(94, 99)])
            array_area_2 = np.concatenate([np.arange(33, 38), np.arange(55, 60), np.arange(77, 82), np.arange(99, 104)])
            array_area_3 = np.concatenate(
                [np.arange(38, 43), np.arange(60, 65), np.arange(82, 87), np.arange(104, 108)])

            # Random initialization of player, fire and citizen
            random_positions_area_0 = np.random.choice(array_area_0, size=8, replace=False)
            random_positions_area_1 = np.random.choice(array_area_1, size=8, replace=False)
            random_positions_area_2 = np.random.choice(array_area_2, size=8, replace=False)
            random_positions_area_3 = np.random.choice(array_area_3, size=8, replace=False)

            obj_1_position = np.concatenate(
                [random_positions_area_0, random_positions_area_1, random_positions_area_2, random_positions_area_3])

            # for i in range(len(obj_1_position)):
            #     tmp_idx = scalar_to_idx(obj_1_position[i])
            #     self.curtain[tmp_idx] = True

            # obj_1_position = np.array([17,18,20,21,23,24,26,
            #                            30, 39,
            #                            43,45,46,48,49,51,52])

            the_plot['P_pos'] = (1, 1)
            the_plot['C_pos'] = [scalar_to_idx(i) for i in obj_1_position]
            the_plot['G_pos'] = (4, 20)

            the_plot['area_0'] = 0
            the_plot['area_1'] = 0
            the_plot['area_2'] = 0
            the_plot['area_3'] = 0

            the_plot['0_steps'] = 0
            the_plot['1_steps'] = 0
            the_plot['2_steps'] = 0
            the_plot['3_steps'] = 0

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if (player_row, player_col) in array_0:
            the_plot['0_steps'] += 1
        elif (player_row, player_col) in array_1:
            the_plot['1_steps'] += 1
        elif (player_row, player_col) in array_2:
            the_plot['2_steps'] += 1
        elif (player_row, player_col) in array_3:
            the_plot['3_steps'] += 1

        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot['terminal_reason'] = 'max_step'
            the_plot.terminate_episode()


class CleanDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(CleanDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            clean_positions = the_plot['C_pos']
            for pos in clean_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if actions == 5 and self.curtain[(player_row - 1, player_col)]:  # grab upward?
            self.curtain[(player_row - 1, player_col)] = False

            #the_plot.add_reward(np.array([1.0, 0.0, 0.0]))
            the_plot.add_reward(np.array([1.0]))

            if (player_row - 1, player_col) in array_0:
                the_plot['area_0'] += 1
            elif (player_row - 1, player_col) in array_1:
                the_plot['area_1'] += 1
            elif (player_row - 1, player_col) in array_2:
                the_plot['area_2'] += 1
            elif (player_row - 1, player_col) in array_3:
                the_plot['area_3'] += 1

            the_plot['steps'] += 1

        if actions == 6 and self.curtain[(player_row + 1, player_col)]:  # grab downward?
            self.curtain[(player_row + 1, player_col)] = False

            the_plot.add_reward(np.array([1.0]))

            if (player_row + 1, player_col) in array_0:
                the_plot['area_0'] += 1
            elif (player_row + 1, player_col) in array_1:
                the_plot['area_1'] += 1
            elif (player_row + 1, player_col) in array_2:
                the_plot['area_2'] += 1
            elif (player_row + 1, player_col) in array_3:
                the_plot['area_3'] += 1

            the_plot['steps'] += 1

        if actions == 7 and self.curtain[(player_row, player_col - 1)]:  # grab leftward?
            self.curtain[(player_row, player_col - 1)] = False

            the_plot.add_reward(np.array([1.0]))

            if (player_row, player_col - 1) in array_0:
                the_plot['area_0'] += 1
            elif (player_row, player_col - 1) in array_1:
                the_plot['area_1'] += 1
            elif (player_row, player_col - 1) in array_2:
                the_plot['area_2'] += 1
            elif (player_row, player_col - 1) in array_3:
                the_plot['area_3'] += 1

            the_plot['steps'] += 1

        if actions == 8 and self.curtain[(player_row, player_col + 1)]:  # grab rightward?
            self.curtain[(player_row, player_col + 1)] = False

            the_plot.add_reward(np.array([1.0]))

            if (player_row, player_col + 1) in array_0:
                the_plot['area_0'] += 1
            elif (player_row, player_col + 1) in array_1:
                the_plot['area_1'] += 1
            elif (player_row, player_col + 1) in array_2:
                the_plot['area_2'] += 1
            elif (player_row, player_col + 1) in array_3:
                the_plot['area_3'] += 1

            the_plot['steps'] += 1


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            goal_position = the_plot['G_pos']
            self.curtain[goal_position] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            self.curtain[(player_row, player_col)] = False
            bonus = (MAX_STEPS - the_plot['steps'])
            # print("max step : ", bonus)
            #the_plot.add_reward(np.array([0.0, 10, 0.0]))
            the_plot.add_reward(np.array([1.0]))
            the_plot['terminal_reason'] = 'terminal_state'
            the_plot.terminate_episode()


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        goal_position = the_plot['G_pos']

        old_current_position = (self.position[0], self.position[1])
        old_distance_to_goal = distance(old_current_position, goal_position)

        if actions == 0:  # go upward?

            self._north(board, the_plot)

            # new_current_position = (self.position[0], self.position[1])
            # new_distance_to_goal = distance(new_current_position, goal_position)
            #
            # if new_distance_to_goal < old_distance_to_goal:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.001]))  # small reward for moving closer to the goal
            # else:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, -0.001]))

        elif actions == 1:  # go downward?
            self._south(board, the_plot)

            # new_current_position = (self.position[0], self.position[1])
            # new_distance_to_goal = distance(new_current_position, goal_position)
            #
            # if new_distance_to_goal < old_distance_to_goal:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.001]))  # small reward for moving closer to the goal
            # else:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, -0.001]))

        elif actions == 2:  # go leftward?
            self._west(board, the_plot)

            # new_current_position = (self.position[0], self.position[1])
            # new_distance_to_goal = distance(new_current_position, goal_position)

            # if new_distance_to_goal < old_distance_to_goal:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.001]))  # small reward for moving closer to the goal
            # else:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, -0.001]))

        elif actions == 3:  # go rightward?
            self._east(board, the_plot)

            # new_current_position = (self.position[0], self.position[1])
            # new_distance_to_goal = distance(new_current_position, goal_position)

            # if new_distance_to_goal < old_distance_to_goal:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, 0.001]))  # small reward for moving closer to the goal
            # else:
            #     the_plot.add_reward(np.array([0.0, 0.0, 0.0, -0.001]))



def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(
        art=LEVEL[2], what_lies_beneath=LEVEL[3], sprites={'P': PlayerSprite},
        drapes={'X': JudgeDrape, 'G': GoalDrape, 'C': CleanDrape},
        update_schedule=[['X'], ['G'], ['C'], ['P']])


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
