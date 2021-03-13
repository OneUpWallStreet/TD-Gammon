# Play a game from start to finish against a random agent(Not even truly random) the opponent gets the list of avilable moves and 
# always selects the first move (if move is possible at that ply). The agent has been able to beat this opponent with score 1 after 
# about 2500/3000 games of self play. After about 7000 games it can beat the agent with score 2 in about 46 moves.


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
            # if count%10 == 0:
                # torch.save(model.state_dict(), 'model.pth')

            #See Who wins Every 100 steps
            fake_1 = copy.deepcopy(env)
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
                lr = 0.07
                
                # self.eligibility_traces[i] = torch.tensor(self.eligibility_traces[i])
                self.eligibility_traces[i] = lamda*self.eligibility_traces[i] + weights.grad

                new_weights = weights + lr*td_error*self.eligibility_traces[i]
                weights.copy_(new_weights)

        return td_error


model = Network()

model.double()

model.load_state_dict(torch.load('model.pth'))

#AI plays 1
#You play -1
env.reset()
print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
print('_________________________________________________________________________')
print(env.board)
print('_________________________________________________________________________')
print('  18  19  20  21  22  23  24  25  26  27')
print('\n')
# time.sleep(2)

TD = TDAgent(model.double())

print('You play First')
action_dict = dict()
step = 1

player =  -1
rolls = env.roll_dice()
fake_board = copy.deepcopy(env.board)
moves = env.all_possible_moves(player,fake_board,rolls)

for i,move in enumerate(moves):
   action_dict[i] =move

# print(action_dict)
# print(moves)

# select = int(input('Select Action:'))
#Alwats do 0 
# action = action_dict[select]
action = action_dict[0]

for a in action:
    reward,done = env.step(a,env.board,player)

while True:

    step +=1
    
    player = env.change_player(player)
    rolls = env.roll_dice()

    fake_board = copy.deepcopy(env.board)
    actions = env.all_possible_moves(player,fake_board,rolls)

    if actions!=None:
        fake = copy.deepcopy(env)
        # print('Here')
        # print(actions)
        best_action,win_prob = TD.select_best_action(actions,fake,player)

        if len(best_action)!=0:
            for a in best_action:
                reward,done = env.step(a,env.board,player)



    print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
    print('_________________________________________________________________________')
    print(env.board)
    print('_________________________________________________________________________')
    print('  18  19  20  21  22  23  24  25  26  27')
    print('\n')
    # time.sleep(2)

    if done:
        print('Winner: ',reward)
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

    print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
    print('_________________________________________________________________________')
    print(env.board)
    print('_________________________________________________________________________')
    print('  18  19  20  21  22  23  24  25  26  27')
    print('\n')
    # time.sleep(2)

    if done:
        print('Winner: ',reward)
        print('Number of Steps to beat random agent:  ',step)
        break

# model = Network()

# model.double()


print(model.output[0].weight)



