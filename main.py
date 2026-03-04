from src.agent import Agent, Player
from src.board import Board
from src.rewards import calculate_reward

#game loop
board = Board()
players = (Player(1), Player(2))

game_running = True

while game_running:
  for player in players:
    print(board.base)
    player.move(board, player.agent)
    if(board.check_win()):
      game_running = False
      break
