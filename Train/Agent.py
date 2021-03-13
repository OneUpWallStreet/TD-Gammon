# This selects the best action based on the value function calculated using Self PLay
# TD-Learning. This does not perfrom any search but with enough computational power
# you could perfrom a 2-ply search, like the orignal TD Gammon. I tried implementing 
# it but it was taking far too long to select a move even with alpha-beta pruning.

import torch
from torch import nn
import numpy
import copy
# from env import BackGammon
import numpy as np

class TDAgent:

    def __init__(self,net):
        self.net = net
        
    def select_best_action(self,actions,env,player):
        best_action = []
        win_prob = 0

        if actions:
            values = [0.0]*len(actions)

            for i,action in enumerate(actions):

                fake_board = copy.deepcopy(env)
                for a in action:
                    _,done = env.step(a,fake_board.board,player)
                features = fake_board.board_features()
                features = torch.tensor(features)
                values[i] = self.net.forward(features)
            if player == 1:
                best_action_index = int(np.argmax(values))
                win_prob = max(values)

            elif player == -1:
                best_action_index = int(np.argmin(values))
                win_prob = min(values)

            best_action = list(actions)[best_action_index]

        return best_action,win_prob