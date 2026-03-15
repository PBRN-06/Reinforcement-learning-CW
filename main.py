from src.agent import *
from src.board import Board
from src.rewards import calculate_reward

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import Qt
import numpy as np
import sys
from typing import Literal

#game loop
board = Board()
players = (Player(), RL_Agent())

game_running = True

class GameWindow(QMainWindow):
  square_size = 64
  margin = 4

  def __init__(self, board : Board, board_size : int, players : tuple[Agent, Agent],
               player_names : tuple[str, str] = ("Player 1", "Player 2")):
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
    self.player_names = player_names

    self.update_game()

  def board_click(self, i, j):
    def click(_, x : int = i, y : int = j):
      if self.awaiting_board_input:
        self.board.update((x,y),self.current_player+1)
        self.awaiting_board_input = False
        self.player_complete()
      else:
        pass
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
      winner_idx = 1 - self.current_player  # player who just moved
      winner_color = "Red" if winner_idx == 0 else "Blue"
      winner_name = self.player_names[winner_idx]
      self._end_game(f"{winner_name} ({winner_color}) wins!")
    elif not np.any(self.board.base == 0):
      self._end_game("It's a draw!")
    else:
      self.update_game()

  def _end_game(self, message: str):
    for btn in self.btns.values():
      btn.setDisabled(True)
    msg = QMessageBox(self)
    msg.setWindowTitle("Game Over")
    msg.setText(message)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()

try:
  app = QApplication()
  window = GameWindow(board, 9, players)

  window.show()
  sys.exit(app.exec())
except Exception as e:
    print(f"Application error: {e}")