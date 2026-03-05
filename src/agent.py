from src.board import Board

from abc import ABC, abstractmethod
from random import randint

class Agent(ABC):
  def __init__(self):
    pass
  
  @abstractmethod
  def command(self, board : Board, reward) -> bool | tuple[int, int]:
    return False #placeholder
    
class Player(Agent):
  def command(self, board : Board, reward) -> bool | tuple[int, int]:
    return False
  
class RL_Agent(Agent):
  def command(self, board : Board, reward) -> bool | tuple[int, int]:
    while(1):
      pos = (randint(0,board.base.shape[0]-1), randint(0,board.base.shape[0]-1))

      if board.base[pos] == 0:
        return pos
    return False


# class Player:
#     def __init__(self, index):
#         self.player_num = index
#         self.agent = Agent()
#         self.set_reward()
#     def set_reward(self):
#         self.reward_struct = 0 #placeholder
#         pass
#     def move(self, board : Board, agent):
#         if agent.command(board, self.reward_struct):
#             print("placeholder")
#         else:
#             #for human play
#             col = int(input("enter column: "))
#             row = int(input("enter row: "))
#             pos = (col, row)
#             board.update(pos, self.player_num)