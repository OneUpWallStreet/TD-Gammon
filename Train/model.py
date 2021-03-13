import time
import torch
from torch import nn
import numpy as np
from env import BackGammon
from Agent import TDAgent
import random
import math
import copy
import time
import copy
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import style
from IPython.display import clear_output 
from itertools import count
env = BackGammon()
count_arr = []
win_arr_random = []

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.style.use('ggplot')

index = count()


# def animate(i):
#     # count_arr.append(next(index))
#     # win_arr_random.append()
#     plt.cla()
#     plt.plot(count_arr, win_arr_random)


def change_player(player):

    if player == 1:
        return -1
    else:
        return 1


def Play_Random_Test(model,env):
    env.reset()
    model.double()

    model.load_state_dict(torch.load('model.pth'))


    #AI plays 1
    #You play -1

    TD = TDAgent(model)

    action_dict = dict()

    player =  -1
    rolls = env.roll_dice()
    fake_board = copy.deepcopy(env.board)
    moves = env.all_possible_moves(player,fake_board,rolls)

    for i,move in enumerate(moves):
        action_dict[i] =move


    #Alwats do 0 
    action = action_dict[0]

    for a in action:
        reward,done = env.step(a,env.board,player)

    while True:
        
        player = env.change_player(player)
        rolls = env.roll_dice()

        fake_board = copy.deepcopy(env.board)
        actions = env.all_possible_moves(player,fake_board,rolls)

        if actions!=None:
            fake = copy.deepcopy(env)
            best_action,win_prob = TD.select_best_action(actions,fake,player)

            if len(best_action)!=0:
                for a in best_action:
                    reward,done = env.step(a,env.board,player)


        if done:
            winner = reward
            break


        player = env.change_player(player)
        rolls = env.roll_dice()

        action_dict = dict()
        fake_board = copy.deepcopy(env.board)
        moves = env.all_possible_moves(player,fake_board,rolls)

        if len(moves)!=0:
            for i,move in enumerate(moves):
                action_dict[i] =move

            #print(action_dict)

            #select = int(input('Select Action:'))

            #Alwats do 0 
            #action = action_dict[select]
            action = action_dict[0]

            for a in action:
                reward,done = env.step(a,env.board,player)


        if done:
            winner = reward
            break

    return reward

# def Play_Agent_VS_Old_Agent(model,env):

#   model = Network()
#   old_model = Network()
  
#   model.double()
#   old_model.double()

#   model.load_state_dict(torch.load('model_new.pth'))

#   old_model.load_state_dict(torch.load('model_old.pth'))


#   TD_new = TDAgent(model)

#   TD_old = TDAgent(old_model)

#   env.reset()
#   player = np.random.choice([-1,1])


#   rolls = env.roll_dice()

#   while True:

#     fake_board = copy.deepcopy(env.board)
#     actions = env.all_possible_moves(player,fake_board,rolls)

#     if actions!=None:
#       fake = copy.deepcopy(env)

#       if player == 1:
#         best_action = TD_new.select_best_action(actions,fake,player)

#       elif player == -1:
#         best_action = TD_old.select_best_action(actions,fake,player)

#       if len(best_action)!=0:
#         for a in best_action:
#           reward,done = env.step(a,env.board,player)

#     if done:
#       winner = reward
#       break

#     player = change_player(player)
#     rolls = env.roll_dice()

#   return winner 



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(198, 80),
            nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        # Pass the input tensor through each of our operations

        x = self.hidden1(x)
        # x = self.hidden2(x)
        x = self.output(x)
        return x

    # def init_weights(self):
    #     for p in self.parameters():
    #         nn.init.zeros_(p)

    def train(self,iters):


        network = self
        
        TD = TDAgent(network)
        count=0
        gen_count_new = 0
        gen_count_old = 0
        
        for eps in range(iters):
            env.reset()
            self.init_eligiblity_trace()

            rolls = env.roll_dice()
            player = env.random_player()

            env.player = player

            

            count+=1
            print("Calculating Weights: {:.5f}".format(count), end="\r")

            step = 0
            #Saving Model Every 10 steps
            if count%100 == 0:
                torch.save(model.state_dict(), 'model.pth')

            #See Who wins Every 100 steps
            fake_1 = copy.deepcopy(env)
            winner_random = Play_Random_Test(self,fake_1)

            # if count%50 ==0:
            #   print('\n Random Agent Match Winner: ',winner_random)
            #   clear_output()

                          
            while True:

                print('\t\t\t\t Working on Step: ',step,end="\r")
                step+=1

                features = env.board_features()
                features = torch.tensor(features)
                p = self.forward(features)
                fake_board = copy.deepcopy(env.board)
                actions = env.all_possible_moves(player,fake_board,rolls)
                # print(actions)
                if actions!=None:
                    fake = copy.deepcopy(env)
                    action,win_prob = TD.select_best_action(actions,fake,player)
                    if len(action)!=0:
                        for a in action:
                            reward,done = env.step(a,env.board,player)
                features = env.board_features()
                features = torch.tensor(features)
                p_next = self.forward(features)

                if done:
                    loss = self.update_weights(p,reward)
                    # print(loss)
                    break
                else:
                    loss = self.update_weights(p,p_next)

                player = env.change_player(player)
                rolls = env.roll_dice()

    
    def play_with_weights(self,p):
        self.init_eligiblity_trace()
        self.zero_grad()
        p.backward()

    def init_eligiblity_trace(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in list(self.parameters())]

    def update_weights(self,p,p_next):

        # self.init_eligiblity_trace()
        self.zero_grad()

        p.backward()

        with torch.no_grad():

            td_error = p_next - p

            parameters = list(self.parameters())

            for i,weights in enumerate(parameters):
    
                lamda = 0.7
                lr = 0.04
                
                # self.eligibility_traces[i] = torch.tensor(self.eligibility_traces[i])
                self.eligibility_traces[i] = lamda*self.eligibility_traces[i] + weights.grad

                new_weights = weights + lr*td_error*self.eligibility_traces[i]
                weights.copy_(new_weights)

        return td_error


model = Network()
model.double()

# model.load_state_dict(torch.load('/Users/Arteezy/Desktop/TD Gammon Final Boss/model_main.pth'))
model.load_state_dict(torch.load('model.pth'))

iters = int(input('Enter Number of Iterations: '))
model.train(iters)

# torch.save(model.state_dict(), 'model_main.pth')

