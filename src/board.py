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