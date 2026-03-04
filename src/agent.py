from src.board import Board

class Agent:
    def __init__(self):
        pass
    def command(self, board : Board, reward):
        return False #placeholder

class Player:
    def __init__(self, index):
        self.player_num = index
        self.agent = Agent()
        self.set_reward()
    def set_reward(self):
        self.reward_struct = 0 #placeholder
        pass
    def move(self, board : Board, agent):
        if agent.command(board, self.reward_struct):
            print("placeholder")
        else:
            #for human play
            col = int(input("enter column: "))
            row = int(input("enter row: "))
            pos = (col, row)
            board.update(pos, self.player_num)