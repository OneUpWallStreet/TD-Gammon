import numpy as np
import time
import copy


class BackGammon:

    def __init__(self):


        self.board = np.zeros(28)

        self.board[1] = -2
        self.board[6] = 5
        self.board[8] = 3
        self.board[12] = -5
        self.board[13] = 5
        self.board[17] = -3
        self.board[19] = -5
        self.board[24] = 2
        self.moves_4 = []
        self.player = 0

        #Board[25] = -1 Goal
        #Board[0] = 1 Goal

        #Board[26] = 1 Bar
        #Board[27] = -1 Bar

        # print(self.board)
        # print(sum(self.board[6:24]>0) == 0)

        # possible_start_pips = np.where(self.board[0:25]>0)
        # print(possible_start_pips[0])

    def reset(self):

        self.board = np.zeros(28)
        self.board[1] = -2
        self.board[6] = 5
        self.board[8] = 3
        self.board[12] = -5
        self.board[13] = 5
        self.board[17] = -3
        self.board[19] = -5
        self.board[24] = 2


    def roll_dice(self):

        d1 = np.random.randint(1,7)
        d2 = np.random.randint(1,7)

        if d1 == d2:
            return [d1]*4

        else:
            return [d1,d2]

    def check_terminal(self):

        if self.board[25] == -15 and self.board[0]>0:
            return -1,True

        elif self.board[0] == 15 and self.board[25]<0:
            return 1,True

        elif self.board[25] == -15 and self.board[0] == 0:
            return -2,True

        elif self.board[0] == 15 and self.board[25] == 0:
            return 2,True

        return 0,False


    # def check_terminal(self):

    #     if self.board[25] == -15:
    #         return -1,True
    #     elif self.board[0] == 15:
    #         return 1,True

    #     return 0,False

    def random_player(self):
        return np.random.choice([-1,1])

    def change_player(self,player):

        if player == 1:
            return -1
        else:
            return 1
        
    def board_features(self):
        
        players = [1,-1]
        features_vector = []
        for player in players:
            for i,point in enumerate(self.board[1:25]):
                if point >0 and  player == 1 or point<0 and player==-1:
                    if point>0:
                        p = 1
                    elif point<0:
                        p = -1

                    if p == player and (point>0 or point<0):
                        if point == 1 or point == -1:
                            features_vector += [1.0, 0.0, 0.0, 0.0]
                        elif point == 2 or point == -2:
                            features_vector += [1.0, 1.0, 0.0, 0.0]
                        elif point >=3 or point <=-3:
                            features_vector += [1.0, 1.0, 1.0, (point - 3.0) / 2.0]
                else:
                    features_vector += [0.0, 0.0, 0.0, 0.0]

            #Number of units on bar and total units on non bar
            if player == 1:
                non_bar_units = 0
                for i in self.board[1:25]:
                    if i>0:
                        non_bar_units +=i
                bar_units = self.board[26]

                features_vector += [bar_units/2.0, non_bar_units/15.0]

            elif player == -1:
                non_bar_units = 0
                for i in self.board[1:25]:
                    if i<0:
                        non_bar_units +=i
                bar_units = self.board[27]

                features_vector += [bar_units/2.0, non_bar_units/15.0]

        if self.player == 1:

            features_vector += [1.0, 0.0]

        elif self.player == -1:
  
            features_vector += [0.0, 1.0]



        assert len(features_vector) == 198, print("Should be 198 instead of {}".format(len(features_vector)))
        return features_vector


    def step(self,move,board,player):

        reward,done = self.check_terminal()
        self.player = player

        if done==True:
            return reward,done

        first_pos = move[0]
        next_pos = move[1]


        if player == 1:
            board[first_pos] -=1

            if board[next_pos]==-1:
                board[27]+=1
                board[next_pos]= 1

            else:
                board[next_pos] +=1

        elif player == -1:
            if first_pos == 27:
                board[first_pos] -=1
            else:
                board[first_pos] +=1

            if board[next_pos]==1:
                board[26]+=1
                board[next_pos] = -1

            else:
                board[next_pos] -=1

        return None,False

    # def possible_moves(self,player,board,rolls):

    #     moves = []

    
    #     if len(rolls)==4:
    #         fake_board_1 = copy.deepcopy(board)
    #         possible_first_moves = self.possible_move(player,fake_board_1,rolls[0])
    #         for first_move in possible_first_moves:
            
    #             _,done = self.step(first_move,fake_board_1,player)
                
    #             if done!=True:
    #                 fake_board_2 = copy.deepcopy(fake_board_1)
    #                 possible_second_moves = self.possible_move(player,fake_board_2,rolls[0])

    #                 for second_move in possible_second_moves:
    #                     print(fake_board_2)
    #                     _,done = self.step(second_move,fake_board_2,player)
    #                     if done!=True:
    #                         fake_board_3 = copy.deepcopy(fake_board_2)
    #                         possible_third_moves = self.possible_move(player,fake_board_3,rolls[0])

    #                         for third_move in possible_third_moves:
    #                             print(fake_board_3)
    #                             _,done = self.step(third_move,fake_board_3,player)
    #                             if done!=True:
    #                                 fake_board_4 = copy.deepcopy(fake_board_3)
    #                                 possible_fourth_moves = self.possible_move(player,fake_board_4,rolls[0])

    #                                 for fourth_move in possible_fourth_moves:
    #                                     print(fake_board_4)
    #                                     _,done = self.step(fourth_move,fake_board_4,player)
    #                                     moves.append(np.array([first_move,second_move,third_move,fourth_move]))
    #                             else:
    #                                 moves.append(np.array([first_move,second_move,third_move]))
    #                     else:
    #                         moves.append(np.array([first_move,second_move]))


                    
    #             else:
    #                 moves.append(np.array([first_move,None]))

    #     return moves


    def all_possible_moves(self,player,board,rolls):
        moves = []
        
        if len(rolls)==2:

            #Using First Dice
            possible_first_moves = self.possible_move(player,board,rolls[0])

            for m1 in possible_first_moves:
                temp_board = copy.deepcopy(board)
                _,done = self.step(m1,temp_board,player)
                possible_second_moves = self.possible_move(player,temp_board,rolls[1])

                for m2 in possible_second_moves:
                    moves.append(np.array([m1,m2]))


            if rolls[0]!=rolls[1]:
                possible_first_moves = self.possible_move(player,board,rolls[1])

                for m1 in possible_first_moves:
                    temp_board = copy.deepcopy(board)
                    _,done = self.step(m1,temp_board,player)
                    possible_second_moves = self.possible_move(player,temp_board,rolls[0])
                    for m2 in possible_second_moves:
                        moves.append(np.array([m1,m2]))


        elif len(rolls)==4:


            possible_first_moves = self.possible_move(player,board,rolls[1])
            
            for m1 in possible_first_moves:

                temp_board_1 = copy.deepcopy(board)

                _,done = self.step(m1,temp_board_1,player)
                possible_second_moves = self.possible_move(player,temp_board_1,rolls[1])


                for m2 in possible_second_moves:
                    temp_board_2 = copy.deepcopy(temp_board_1)
                    _,done = self.step(m2,temp_board_2,player)
                    possible_third_moves = self.possible_move(player,temp_board_2,rolls[1])


                    for m3 in possible_third_moves:
                        temp_board_3 = copy.deepcopy(temp_board_2)
                        _,done = self.step(m3,temp_board_3,player)
                        possible_fourth_moves = self.possible_move(player,temp_board_3,rolls[1])

                        for m4 in possible_fourth_moves:
                            # print(temp_board_3)
                            # time.sleep(2)
                            moves.append(np.array([m1,m2,m3,m4]))

        #Double Move not working 
        if len(moves) == 0:

            #Dice 1
            possible_first_moves = self.possible_move(player,board,rolls[0])
            for m1 in possible_first_moves:
                moves.append(np.array([m1]))

            #Dice 2
            possible_first_moves = self.possible_move(player,board,rolls[1])
            for m1 in possible_first_moves:
                moves.append(np.array([m1]))

        return moves

    def possible_move(self,player,board,die):

        possible_moves = []
        terminal_moves = []
        bar_moves = []
        

        if player == 1:

            #Checking if bar move is possible
            if board[26]>0:
                pos = 25-die
                if board[pos]>-2:
                    possible_moves.append(np.array([26,pos]))
                    bar_moves.append(np.array([26,pos]))

            else:
                if len(np.where(board[7:25]>0)[0])==0:
                    if board[die]>0:
                        possible_moves.append(np.array([die,0]))
                        terminal_moves.append(np.array([die,0]))

                
                possible_start_pos = np.where(board[1:25]>0)[0]

                for i,p in enumerate(possible_start_pos):
                    possible_start_pos[i] +=1


                for start_pos in possible_start_pos:

                    next_pos = start_pos - die
                    if next_pos>0:
                        if board[next_pos]>-2:

                            possible_moves.append(np.array([start_pos,next_pos]))
                            terminal_moves.append(np.array([start_pos,next_pos]))

                    # if len(np.where(board[7:25]>0)[0])==0 and next_pos==0:




                        
            if not bar_moves and len(np.where(board[7:25]>0)[0])==0 and board[26]==0:
                # print('Nope')
                possible_start_pos = np.where(board[1:25]>0)[0]
                
                for i,p in enumerate(possible_start_pos):
                    possible_start_pos[i] +=1

                for start_pos in possible_start_pos:

                    if die-start_pos==0:
                        possible_moves.append(np.array([start_pos,0]))
                        

                    elif die-start_pos>0 and len(terminal_moves)==0:
                        possible_moves.append(np.array([start_pos,0]))


        if player == -1:

            if board[27]>0:
                pos = die
                if board[pos]<2:
                    possible_moves.append(np.array([27,pos]))
                    bar_moves.append(np.array([27,pos]))

            else:

                if len(np.where(board[0:19]<0)[0])==0:

                    if board[25-die]<0:
                        possible_moves.append(np.array([25-die,25]))
                        terminal_moves.append(np.array([25-die,25]))

                possible_start_pos = np.where(board[0:25]<0)[0]
                # print(possible_start_pos)

                for start_pos in possible_start_pos:

                    next_pos = start_pos + die
                    if next_pos <25:
                        if board[next_pos]<2:
                            possible_moves.append(np.array([start_pos,next_pos]))
                            terminal_moves.append(np.array([start_pos,next_pos]))

            if not bar_moves and len(np.where(board[0:19]<0)[0])==0 and board[27]==0:

                possible_start_pos = np.where(board[0:25]<0)[0]
                # print(possible_start_pos)


                for start_pos in possible_start_pos:

                    if die == (25-start_pos):
                        possible_moves.append(np.array([start_pos,25]))
                    
                    elif die >(25 - start_pos) and len(terminal_moves)==0:
                        possible_moves.append(np.array([start_pos,25]))

                    
        return possible_moves
