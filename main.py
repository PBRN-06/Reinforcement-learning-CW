from src.agent import Agent, Player, RL_Agent
from src.board import Board
from src.rewards import calculate_reward

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt
import sys
from typing import Literal

# Game: human (Player) vs agent. Use RL_Agent for random moves.
# To use a trained gomoku_rl policy: from src.agent import TrainedGomokuAgent
# and set players = (Player(), TrainedGomokuAgent("checkpoint_iter_100.pt"))
board = Board()
players = (Player(), RL_Agent())

game_running = True

class GameWindow(QMainWindow):
  square_size = 64
  margin = 4

  def __init__(self, board : Board, board_size : int, players : tuple[Agent, Agent]):
    super().__init__()

    self.setGeometry(100, 100, 1024, 768)
    self.setWindowTitle("Gomoku")
    central_widget = QWidget()
    sz = (GameWindow.square_size + GameWindow.margin) * board_size - GameWindow.margin
    central_widget.setFixedSize(sz, sz)

    self.board_render = QGridLayout(central_widget)
    self.setLayout(self.board_render)
    self.btns : dict[tuple[int, int], QPushButton] = dict()

    for i in range(board_size):
      for j in range(board_size):
        btn = QPushButton()
        btn.setFixedSize(GameWindow.square_size,GameWindow.square_size)
        btn.clicked.connect(self.board_click(i,j))

        self.board_render.addWidget(btn, i, j)
        self.btns[(i,j)] = btn
    
    self.setCentralWidget(central_widget) 



    self.board = board
    self.board_sz = board_size
    self.awaiting_board_input = False
    self.current_player : Literal[0,1] = 0
    self.players = players

    self.update_game()

  def board_click(self, i, j):
    def click(_, x : int = i, y : int = j):
      if self.awaiting_board_input:
        self.board.update((x,y),self.current_player+1)
        self.awaiting_board_input = False
        self.player_complete()
      else:
        print("Not expecting input")
    return click
  
  def update_game(self):
    cmd = self.players[self.current_player].command(
      self.board, calculate_reward(board.base, self.current_player + 1))
    if cmd:
      self.board.update(cmd, self.current_player + 1)
      self.player_complete()
    else:
      self.awaiting_board_input = True

  def player_complete(self):
    # Update board render
    for i in range(self.board_sz):
      for j in range(self.board_sz):
        if self.board.base[i,j] == 0: continue
        if not self.btns[(i,j)].isEnabled(): continue
        
        self.btns[(i,j)].setStyleSheet(
          "QPushButton { background-color: " + 
          ("red;\n" if self.current_player == 0 else "blue;\n") +
          """
                color: black;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {background-color: grey;}
            QPushButton:pressed {background-color: grey;}""")
        self.btns[(i,j)].setDisabled(True)
        
    self.current_player = (self.current_player + 1) % 2
    if self.board.check_win():
      self.close()
    else:
      self.update_game()

try:
  app = QApplication()
  window = GameWindow(board, 9, players)

  window.show()
  sys.exit(app.exec())
except Exception as e:
    print(f"Application error: {e}")