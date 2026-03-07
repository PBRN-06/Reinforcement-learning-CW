import numpy as np

def calculate_reward(board : np.ndarray, player, board_size=None):
  """
  Calculate reward for a provided game state.

  Args:
    board: 2D numpy array where 0=empty, 1=self, 2=opponent
    player: Current player (1 or 2). Board will be normalized so player sees themselves as 1
    board_size: Size of the board (will be inferred if not provided)

  Returns:
    float: Total reward value
  """
  if board_size is None:
    board_size = board.shape[0]

  # Normalize board so current player is always 1
  if player == 2:
    normalized_board = np.where(board == 0, 0, np.where(board == 1, 2, 1))
  else:
    normalized_board = board.copy()

  # Reward weights for different sequence lengths
  weights = {
    # Lenth : Reward
    2:        1,
    3:        5,
    4:        50,
    5:        0 # Used purely for detection - value is encoded in standard alpha-zero
  }

  # Penalty weights for opponent sequences (negative)
  penalty_weights = {
    # Lenth : Reward
    2:        -3,
    3:        -10,
    4:        -80,
    5:        0
  }

  total_reward = 0

  player_sequences = count_sequences(normalized_board, 1, board_size)
  opponent_sequences = count_sequences(normalized_board, 2, board_size)

  for length, count in player_sequences.items():
    total_reward += weights[length] * count
  for length, count in opponent_sequences.items():
    total_reward += penalty_weights[length] * count

  return total_reward


def count_sequences(board : np.ndarray, player_id, board_size):
  """
  Count all sequences of length 2, 3, 4, and 5 for a given player.

  Args:
    board: 2D numpy array
    player_id: Player to count sequences for (1 or 2)
    board_size: Size of the board

  Returns:
    dict: {length: count} for lengths 2-5
  """
  sequences = {2: 0, 3: 0, 4: 0, 5: 0}

  directions = [
      (0, 1),   # Horizontal
      (1, 0),   # Vertical
      (1, 1),   # Diagonal \
      (1, -1)   # Diagonal /
  ]

  for row in range(board_size):
    for col in range(board_size):
      if board[row, col] != player_id:
        continue
      
      for dr, dc in directions:
        # Ignore if there is a marker "before" this one (i.e. reverse direction)
        # Prevents double counting
        before_row = row - dr
        before_col = col - dc
        has_before = (0 <= before_row < board_size and 0 <= before_col < board_size and
                      board[before_row, before_col] == player_id)
        if(has_before):
          continue
        
        for length in range(5, 1, -1):
          if is_valid_sequence(board, row, col, dr, dc, player_id, length, board_size):
            sequences[length] += 1
            # Can immediately break since iterating backward finds the longest first, 
            #  further results are ignored anyways
            break 

  return sequences


def is_valid_sequence(board : np.ndarray, start_row, start_col, dr, dc, player_id, length, board_size):
  """
  Check if there's a valid sequence of given length starting from position.
  A valid sequence has exactly 'length' consecutive pieces with at least one open end.

  Args:
    board: 2D numpy array
    start_row, start_col: Starting position
    dr, dc: Direction increments
    player_id: Player to check for
    length: Desired sequence length
    board_size: Size of the board

  Returns:
    bool: True if valid sequence exists
  """
  # Check if sequence fits
  end_row = start_row + (length - 1) * dr
  end_col = start_col + (length - 1) * dc

  if not (0 <= end_row < board_size and 0 <= end_col < board_size):
    return False

  # Check if all positions in sequence belong to player
  for i in range(length):
    row = start_row + i * dr
    col = start_col + i * dc
    if board[row, col] != player_id:
      return False
    
  # Check either beginning or end are exposed to allow growth if lenth < 5
  # Won't trigger if our own marker is there since before is checked before 
  #  this functions is called, and these are checked from longest to shortest
  #  so after can't trigger either
  #
  # Incorporating this should help further disuade finding 3s and 4s which are 
  # actually blocked
  #
  # Still counts 5s both for win detection purposes and since those should be
  # rewarded regardless
  if length < 5:
    pre_row = start_row - dr; pre_col = start_col - dc
    post_row = start_row + length * dr; post_col = start_col + length * dc

    # Check if positions are in bounds before accessing
    pre_in_bounds = (0 <= pre_row < board_size and 0 <= pre_col < board_size)
    post_in_bounds = (0 <= post_row < board_size and 0 <= post_col < board_size)

    # Get values (0 if out of bounds = open end)
    pre_val = board[pre_row, pre_col] if pre_in_bounds else 0
    post_val = board[post_row, post_col] if post_in_bounds else 0

    # Both ends blocked = invalid sequence
    if pre_val != 0 and post_val != 0:
      return False

  return True