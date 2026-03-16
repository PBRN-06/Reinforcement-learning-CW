import numpy as np
from src.rewards import count_sequences

class Board:
    def __init__(self, size=9):
        self.base = np.zeros([size, size])

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
        seqs = count_sequences(self.base, 1, self.base.shape[0])
        if seqs[5] >= 1:
          print("Player 1 wins")
          return True
        seqs = count_sequences(self.base, 2, self.base.shape[0])
        if seqs[5] >= 1:
          print("Player 2 wins")
          return True
        
        return False