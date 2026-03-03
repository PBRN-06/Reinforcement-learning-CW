import numpy as np

class Board:
    def __init__(self):
        self.base = np.zeros([9,9])

    def update(self, pos, index):
        if self.check_valid(pos):
            self.base[pos] = index
        else:
            print("invalid move, space not empty")
    def check_valid(self, pos):
        if self.base[pos] == 0:
            return True
        else:
            return False
    def check_win(self):
        pass    
    


class Agent:
    def __init__(self):
        pass
    def command(self, board, reward):
        return False #placeholder

class Player:
    def __init__(self, index):
        self.player_num = index
        self.agent = Agent()
        self.set_reward()
    def set_reward(self):
        self.reward_struct = 0 #placeholder
        pass
    def move(self, board, agent):
        if agent.command(board, self.reward_struct):
            print("placeholder")
        else:
            #for human play
            col = int(input("enter column: "))
            row = int(input("enter row: "))
            pos = (col, row)
            board.update(pos, self.player_num)

#game loop
board = Board()
players = (Player(1), Player(2))

while True:
    for player in players:
        print(board.base)
        player.move(board, player.agent)
        board.check_win()
