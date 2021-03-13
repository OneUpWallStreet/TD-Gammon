# Terminal Interface to play against human, the player has to play first. Doubling is not allowed. Every 
# move the agent makes it estimates the probablity of winning and plots it. When it expects that probablity
# of winning is less than 5% it will ask if it can resgin. If you input 'Y' or 'y', you win with score -1 
# or -2. Every time agent moves it updates graph of win probablity.


import time
import torch
from torch import nn
import numpy as np
from env import BackGammon
from Agent import TDAgent
import random
import math
import sys
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
plt.style.use('ggplot')

# plt.style.use('seaborn-white')
# import seaborn as sns
# sns.set()

# from model import Network
env = BackGammon()
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

    def init_weights(self):
        for p in self.parameters():
            nn.init.zeros_(p)

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
            # if count%1000 == 0:
                # torch.save(model.state_dict(), 'model.pth')

            #See Who wins Every 100 steps
            # fake_1 = copy.deepcopy(env)
            # winner_random = Play_Random_Test(self,fake_1)

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
                lr = 0.1
                
                # self.eligibility_traces[i] = torch.tensor(self.eligibility_traces[i])
                self.eligibility_traces[i] = lamda*self.eligibility_traces[i] + weights.grad

                new_weights = weights + lr*td_error*self.eligibility_traces[i]
                weights.copy_(new_weights)

        return td_error


def change_player(player):

    if player == 1:
        return -1
    else:
        return 1

def input_dice_rolls():

    d1 = int(input('Enter Dice 1: '))

    d2 = int(input('Enter Dice 2: '))

    if d1 == d2:
        return [d1]*4
    else:
        return [d1,d2]

model = Network()

count_num = 0
count_arr = []

value_arr = []

model.double()

model.load_state_dict(torch.load('model.pth'))

TD = TDAgent(model)

print('Starting Game,-1 plays First')
env.reset()
player = -1
rolls = input_dice_rolls()
print('\n')
print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
print('_________________________________________________________________________')
print(env.board)
print('_________________________________________________________________________')
print('  18  19  20  21  22  23  24  25  26  27')
print('\n')
print('Player moves -1')
print('\n')
print('Rolls: ',rolls)

fake_board = copy.deepcopy(env.board)
actions = env.all_possible_moves(player,fake_board,rolls)

print('Enter %s moves if possible: '%(len(rolls)))

rolls_copy = rolls
for x in range(len(rolls)):

    print('Available Dice Rolls: ',rolls_copy)

    start_pos = int(input('Start Position: '))
    end_pos = int(input('End Position '))

    action  = [start_pos,end_pos]

    _,done = env.step(action,env.board,player)

    print('\n')
    print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
    print('_________________________________________________________________________')
    print(env.board)
    print('_________________________________________________________________________')
    print('  18  19  20  21  22  23  24  25  26  27')
    print('\n')

    if start_pos == 27 or start_pos == 26:
        rolls_copy.remove(end_pos)

    else:
        rolls_copy.remove(np.abs(start_pos-end_pos))


while True:

    #If you are playing with agent while its training than you can load the latest 
    # batch while in this loop
    # model.load_state_dict(torch.load('model.pth'))
    # TD = TDAgent(model)

    player = env.change_player(player)
    rolls = input_dice_rolls()

    fake_board = copy.deepcopy(env.board)
    actions = env.all_possible_moves(player,fake_board,rolls)

    if actions!=None:
        fake = copy.deepcopy(env)
        # print('Here')
        # print(actions)
        best_action,win_prob = TD.select_best_action(actions,fake,player)
        

        if len(best_action)!=0:
            print('AI Expects that the Probablity of Winning is: %s'%(win_prob.item()))

            if win_prob<=0.05:
                surrender = input('Agent wants to Surrender(Y/N)')

                if surrender == 'y' or surrender == 'Y':
                    print('Good Game Well PLayed')
                    time.sleep(5)
                    break

            value_arr.append(win_prob)
            count_arr.append(count_num)
            count_num+=1
            for a in best_action:
                print('AI Taking Action: ',a)

                
                reward,done = env.step(a,env.board,player)
                time.sleep(1)
                

    print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
    print('_________________________________________________________________________')
    print(env.board)
    print('_________________________________________________________________________')
    print('  18  19  20  21  22  23  24  25  26  27')
    print('\n')

    if done:
        print('Winner is: ',reward)
        break

    plt.figure(figsize=(11,5))
    plt.plot(count_arr,value_arr)
    plt.xlabel("Step")
    plt.ylabel("Win Probablity")
    plt.title("Current Value")
    plt.ylim(ymin=0) 
    plt.ylim(ymax=1)
    plt.show(block=False)
    # plt.pause(7)
    # plt.close()



    rolls = input_dice_rolls()
    player = env.change_player(player)

    fake_board = copy.deepcopy(env.board)
    actions = env.all_possible_moves(player,fake_board,rolls)



    if len(actions)!=0:
        print('Available Actions: ',len(actions))

        if len(actions)<=10:
            print('\n')
            print('Available Actions...')
            print(actions)
            print('\n')

        rolls_copy = rolls

        for x in range(len(rolls)):
            print('Available Dice Rolls: ',rolls_copy)

            start_pos = int(input('Start Position: '))
            end_pos = int(input('End Position '))
            used_roll = int(input('Enter Used Roll: '))

            action  = [start_pos,end_pos]

            reward,done = env.step(action,env.board,player)

            print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
            print('_________________________________________________________________________')
            print(env.board)
            print('_________________________________________________________________________')
            print('  18  19  20  21  22  23  24  25  26  27')
            print('\n')

            if done:
                print('Winner is: ',reward)
                break

            rolls_copy.remove(used_roll)
            
            # if start_pos == 27:
            #     rolls_copy.remove(end_pos)

            # elif end_pos == 25:

            #     if end_pos-start_pos == used_roll:
            #         rolls_copy.remove(used_roll)                

            # else:
            #     rolls_copy.remove(np.abs(start_pos-end_pos))

            if len(rolls_copy) == 0:
                break

            die = rolls_copy[0]
            fake_board = copy.deepcopy(env.board)
            spec_action = env.possible_move(player,fake_board,die)

            if len(spec_action) == 0:
                break

    else:
        print('No Possible Moves for Player...')

    if done:
        print('Winner is: ',reward)
        break
