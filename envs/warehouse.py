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

MAX_STEPS = 400
LEVEL = [
    ['################################',
     '#PAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '#AAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '#AAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '#AAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '#AAAAAAAAAAAAAAABBBBBBBBBBBBBBB#',
     '################################'],

    ['################################',
     '#A                             #',
     '#                              #',
     '#                              #',
     '#                              #',
     '#                              #',
     '################################']
]



COLOR = {'#': (428, 135, 0),
         'P': (388, 400, 999),
         'A': (870, 838, 678),
         'B': (900, 0, 20),
         'G' :(0, 600, 67),
         'C':(900, 300, 900)
         }


def scalar_to_idx(x):
    row = x // 32
    col = x % 32
    return row, col

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

            array_area_1 = np.concatenate([np.arange(34, 43), np.arange(65, 75), np.arange(97, 107), np.arange(129, 139), np.arange(161, 171)])
            array_area_2 = np.concatenate([np.arange(43, 53), np.arange(75, 85), np.arange(107, 117), np.arange(139, 149), np.arange(171, 181)])
            array_area_3 = np.concatenate([np.arange(53, 63), np.arange(85, 95), np.arange(117, 127),  np.arange(149, 159), np.arange(181, 191)])

            # Random initialization of player, fire and citizen
            random_positions_area_1 = np.random.choice(array_area_1, size=20, replace=False)
            random_positions_area_2 = np.random.choice(array_area_2, size=20, replace=False)
            random_positions_area_3 = np.random.choice(array_area_3, size=20, replace=False)

            obj_1_position_area_1 = random_positions_area_1[:10]
            obj_2_position_area_1 = random_positions_area_1[10:]

            obj_1_position_area_2 = random_positions_area_2[:10]
            obj_2_position_area_2 = random_positions_area_2[10:]

            obj_1_position_area_3 = random_positions_area_3[:10]
            obj_2_position_area_3 = random_positions_area_3[10:]


            obj_1_position = np.concatenate([obj_1_position_area_1, obj_1_position_area_2, obj_1_position_area_3])
            obj_2_position = np.concatenate([obj_2_position_area_1, obj_2_position_area_2, obj_2_position_area_3])

            random_positions = np.concatenate([obj_1_position, obj_2_position]) # total 60
            for i in range(len(random_positions)):
                tmp_idx = scalar_to_idx(random_positions[i])
                self.curtain[tmp_idx] = True
            the_plot['P_pos'] = (1, 1)
            the_plot['G_pos'] = [scalar_to_idx(i) for i in random_positions[:30]]
            the_plot['C_pos'] = [scalar_to_idx(i) for i in random_positions[30:]]


        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 9) or (self._step_counter == self._max_steps):
            the_plot.terminate_episode()



class Obj1(plab_things.Drape):
    def __init__(self, curtain, character):
        super(Obj1, self).__init__(curtain, character)

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
        if actions == 5 and self.curtain[(player_row-1, player_col)]: # grab upward?
            self.curtain[(player_row-1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))
        if actions == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
            self.curtain[(player_row+1, player_col)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))
        if actions == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
            self.curtain[(player_row, player_col-1)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))
        if actions == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
            self.curtain[(player_row, player_col+1)] = False
            the_plot.add_reward(np.array([1.0, 0.0]))


class Obj2(plab_things.Drape):
    def __init__(self, curtain, character):
        super(Obj2, self).__init__(curtain, character)

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
        if actions == 5 and self.curtain[(player_row-1, player_col)]: # grab upward?
            self.curtain[(player_row-1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))
        if actions == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
            self.curtain[(player_row+1, player_col)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))
        if actions == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
            self.curtain[(player_row, player_col-1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))
        if actions == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
            self.curtain[(player_row, player_col+1)] = False
            the_plot.add_reward(np.array([0.0, 1.0]))


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
    return ascii_art.ascii_art_to_game(
        art=LEVEL[0], what_lies_beneath=LEVEL[1], sprites={'P': PlayerSprite}, drapes = {'X': JudgeDrape, 'G': Obj1, 'C':Obj2},
        update_schedule=[['X'],['G'],['C'],['P']])

def main(demo):
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
