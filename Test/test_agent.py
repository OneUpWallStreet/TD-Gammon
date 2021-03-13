# BackGammon Testing Enviroment of the main Agent trained over around 25000 games vs a Agent trained on the Google Colab 
# notebook for about 6000 games. As you increase the number of self play games on the main Agent the frequency with which
# it gammons the Google Colab Agent increases. (By Frequency I mean by winning with a score of 2). Winning with 2 means that 
# it was able to beat the opponent before it was able to bear off a single checker.



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

def change_player(player):

    if player == 1:
        return -1
    else:
        return 1

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

            # step = 0
            #Saving Model Every 10 steps
            # if count%10 == 0:
                # torch.save(model.state_dict(), 'model_new_main.pth')

            #See Who wins Every 100 steps
            # fake_1 = copy.deepcopy(env)
            # winner_random = Play_Random_Test(self,fake_1)

            # if count%50 ==0:
            #   print('\n Random Agent Match Winner: ',winner_random)
            #   clear_output()

                          
            while True:

                print('\t\t\t\t Working on Step: ',step,end="\r")
                # step+=1

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


def Play_Agent_VS_Old_Agent(env):
  
  model = Network()
  step = 0
  old_model = Network()
  
  model.double()
  old_model.double()

  print('Model_New_Main is 1 ')
  print('Model_colab_old is -1')
   
  model.load_state_dict(torch.load('model.pth'))

  old_model.load_state_dict(torch.load('model_weak.pth'))

  TD_new = TDAgent(model)

  TD_old = TDAgent(old_model)

  env.reset()
  player = np.random.choice([-1,1])


  rolls = env.roll_dice()

  while True:
    step+=1
    print('Player: ',player)
    print('Rolls: ', rolls)
    print('  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17')
    print('_________________________________________________________________________')
    print(env.board)
    print('_________________________________________________________________________')
    print('  18  19  20  21  22  23  24  25  26  27')
    print('\n')

    fake_board = copy.deepcopy(env.board)
    actions = env.all_possible_moves(player,fake_board,rolls)

    if actions!=None:
      fake = copy.deepcopy(env)

      if player == 1:
        best_action,win_prob = TD_new.select_best_action(actions,fake,player)

      elif player == -1:
        best_action,win_prob = TD_old.select_best_action(actions,fake,player)

      if len(best_action)!=0:
        for a in best_action:
          reward,done = env.step(a,env.board,player)

    if done:
      print('Won in %s moves!'%(step))
      winner = reward
      break

    player = change_player(player)
    rolls = env.roll_dice()

  return winner 


win = Play_Agent_VS_Old_Agent(env)
# print('Winner is...')
# time.sleep(2)

print('Winner of Macbook Agent vs Google Colab Agent is: ',win)


model = Network()
old_model = Network()

model.double()
old_model.double()
model.load_state_dict(torch.load('model.pth'))
old_model.load_state_dict(torch.load('model_weak.pth'))


print('Mac Weights')

print(model.hidden1[0].weight)

print('Colab Weights')

print(old_model.hidden1[0].weight)