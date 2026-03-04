from src.agent import Agent, Player
from src.board import Board  

#game loop
board = Board()
players = (Player(1), Player(2))

while True:
    for player in players:
        print(board.base)
        player.move(board, player.agent)
        board.check_win()
