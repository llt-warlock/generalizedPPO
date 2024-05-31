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

MAX_STEPS = 30
LEVEL = [

    ['######################',  # row 6
     '#P$$$$$$$$$&&&&&&&&&&#',  # column 20
     '#$$$$$$$$$$&&&&&&&&&&#',
     '#$$$$$$$$$$&&&&&&&&&&#',
     '#$$$$$$$$$$&&&&&&&&&&#',
     '#$$$$$$$$$$&&&&&&&&&&#',
     '#$$$$$$$$$$&&&&&&&&&&#',
     '######################'],



    ['######################',
     '#$                   #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '#                    #',
     '######################']
]



COLOR = {'#': (428, 135, 0),
         'P': (388, 400, 999),
         '$':(870, 838, 678),
         '&':(900, 0, 20),
         'G' :(0, 600, 67),
         'C':(900, 300, 900)
         }


def scalar_to_idx(x):
    row = x%6
    col = x%20
    return (row+1, col+1)

# class region_1_Drape(plab_things.Drape):
#     def __init__(self, curtain, character):
#         super(GoalDrape, self).__init__(curtain, character)
#
#     def update(self, actions, board, layers, backdrop, things, the_plot):
#         del backdrop
#
#         if the_plot.frame == 0:
#             self.curtain[scalar_to_idx(23)] = True


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

            # region information
            region_1 = np.where(board == ord('$'))
            region_2 = np.where(board == ord('&'))

            temp_region_1 = []
            for i in range(len(region_1[0])):
                temp_region_1.append((region_1[0][i], region_1[1][i]))

            temp_region_1.append((1, 1))
            the_plot['$'] = temp_region_1

            temp_region_2 = []
            for i in range(len(region_2[0])):
                temp_region_2.append((region_2[0][i], region_2[1][i]))

            the_plot['&'] = temp_region_2

            # objective 1
            array_region_1 = np.concatenate([np.arange(1, 10), np.arange(20, 30), np.arange(40, 50), np.arange(60, 70), np.arange(80, 90), np.arange(100, 110)])
            array_region_2 = np.concatenate([np.arange(10, 20), np.arange(30, 40), np.arange(50, 60),  np.arange(70, 80), np.arange(90, 100), np.arange(110, 120)])

            # Random initialization of player, fire and citizen
            random_positions_region_1 = np.random.choice(array_region_1, size=12, replace=False)
            random_positions_region_2 = np.random.choice(array_region_2, size=12, replace=False)

            objective_1_position_region_1 = random_positions_region_1[:6]
            objective_1_position_region_2 = random_positions_region_2[:6]
            objective_2_position_region_1 = random_positions_region_1[6:]
            objective_2_position_region_2 = random_positions_region_2[6:]
            objective_1_position = np.concatenate([objective_1_position_region_1, objective_1_position_region_2])
            objective_2_position = np.concatenate([objective_2_position_region_1, objective_2_position_region_2])
            random_positions = np.concatenate([objective_1_position, objective_2_position])
            for i in range(len(random_positions)):
                tmp_idx = scalar_to_idx(random_positions[i])
                self.curtain[tmp_idx] = True
            the_plot['P_pos'] = (1, 1)
            the_plot['G_pos'] = [scalar_to_idx(i) for i in random_positions[:12]]
            the_plot['C_pos'] = [scalar_to_idx(i) for i in random_positions[12:]]


        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot.terminate_episode()



class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop


        if the_plot.frame == 0:
            goal_1_positions = the_plot['G_pos']
            for pos in goal_1_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        #print("in goal : ", player_row, "  ", player_col)


        # Check for 'pick up' action:
        if self.curtain[(player_row, player_col)]:  # grab upward?
            self.curtain[(player_row, player_col)] = False
            the_plot.add_reward(np.array([1., 0.]))


class CitizenDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(CitizenDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            goal_2_positions = the_plot['C_pos']
            for pos in goal_2_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Check for 'pick up' action:
        if self.curtain[(player_row, player_col)]:  # grab upward?
            self.curtain[(player_row, player_col)] = False
            the_plot.add_reward(np.array([0., 1.]))


        # if self.curtain[player_pattern_position]:
        #    the_plot.add_reward(np.array([1, 0]))
        #    self.curtain[player_pattern_position] = False


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

        the_plot['P_pos'] = (self.position.row, self.position.col)
        #print("p post : ", the_plot['P_pos'])

def make_game(seed=None, demo=False):
    return ascii_art.ascii_art_to_game(
        art=LEVEL[0], what_lies_beneath=LEVEL[1], sprites={'P': PlayerSprite}, drapes = {'X': JudgeDrape, 'G': GoalDrape, 'C':CitizenDrape},
        update_schedule=[['X'],['G'],['C'],['P']])

def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)

    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         -1: 4,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=COLOR)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    main(demo=False)
