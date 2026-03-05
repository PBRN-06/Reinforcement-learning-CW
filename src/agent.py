from src.board import Board

from abc import ABC, abstractmethod
from random import randint


class Agent(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def command(self, board: Board, reward) -> bool | tuple[int, int]:
    return False  # placeholder


class Player(Agent):
  def command(self, board: Board, reward) -> bool | tuple[int, int]:
    return False


class RL_Agent(Agent):
  def command(self, board: Board, reward) -> bool | tuple[int, int]:
    while True:
      pos = (randint(0, board.base.shape[0] - 1), randint(0, board.base.shape[0] - 1))
      if board.base[pos] == 0:
        return pos
    return False


class TrainedGomokuAgent(Agent):
  """
  Agent that uses a trained gomoku_rl policy (CNN). Same interface as RL_Agent.
  Requires: gomoku_rl, torch. Pass checkpoint path to load weights.
  """
  def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
    super().__init__()
    self._checkpoint_path = checkpoint_path
    self._device_str = device
    self._policy_net = None
    self._device = None

  def _load_model(self):
    if self._policy_net is not None:
      return
    import torch
    from gomoku_rl.models.policy_net import GomokuPolicyNet
    from gomoku_rl.env.board import GomokuBoard
    self._GomokuPolicyNet = GomokuPolicyNet
    self._GomokuBoard = GomokuBoard
    self._device = torch.device(self._device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    self._policy_net = self._GomokuPolicyNet().to(self._device)
    if self._checkpoint_path:
      try:
        state = torch.load(self._checkpoint_path, map_location=self._device, weights_only=True)
      except TypeError:
        state = torch.load(self._checkpoint_path, map_location=self._device)
      if "policy_net" in state:
        self._policy_net.load_state_dict(state["policy_net"])
      else:
        self._policy_net.load_state_dict(state)

  def command(self, board: Board, reward) -> bool | tuple[int, int]:
    self._load_model()
    state = self._GomokuBoard.state_from_repo_board(board.base)
    legal = [(r, c) for r in range(board.base.shape[0]) for c in range(board.base.shape[1]) if board.base[r, c] == 0]
    if not legal:
      return False
    row, col = self._policy_net.predict(state, legal_actions=legal, deterministic=True, device=self._device)
    return (row, col)


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