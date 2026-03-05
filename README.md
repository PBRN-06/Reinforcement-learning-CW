# Reinforcement-learning-CW

Gomoku (9×9, five-in-a-row) with a PySide6 GUI and an optional deep RL agent. Play as human vs random or vs a trained CNN policy.

## Project structure

```
Reinforcement-learning-CW/
├── main.py                 # GUI: human vs agent (PySide6)
├── src/
│   ├── board.py            # 9×9 board (0=empty, 1=P1, 2=P2), update, check_win
│   ├── agent.py            # Agent, Player, RL_Agent (random), TrainedGomokuAgent
│   └── rewards.py          # calculate_reward, count_sequences (heuristic eval)
├── gomoku_rl/              # Deep RL training pipeline
│   ├── config.py           # Board size, CNN/training/reward/MCTS hyperparameters
│   ├── train.py            # CLI: self-play → PPO → metrics, checkpoints
│   ├── env/
│   │   ├── board.py        # GomokuBoard (same 0/1/2), step(), to_state_tensor(), legal_actions()
│   │   └── reward_wrapper.py  # Sparse win/loss/draw + pattern bonuses with decay
│   ├── models/
│   │   ├── policy_net.py   # CNN Actor-Critic, predict(board_state) → (row, col)
│   │   └── mcts.py         # Optional MCTS using policy prior and value
│   └── training/
│       ├── self_play.py    # Model vs model trajectory collection
│       ├── train_step.py   # PPO (clipped policy, value loss, entropy, GAE)
│       └── metrics.py     # Win ratio P1/P2, turns per game (sliding window)
├── requirements.txt
└── README.md
```

## Dependencies

- **GUI and repo logic:** `numpy`, `pyside6` (see `requirements.txt`).
- **Training (gomoku_rl):** `torch`. Install with `pip install torch` (or use your env).

## Run the game (GUI)

From the project root:

```bash
pip install -r requirements.txt
python main.py
```

- **Player 1:** Human (click a cell when it’s your turn).
- **Player 2:** By default the random agent (`RL_Agent`). To use a trained policy, edit `main.py`:

```python
from src.agent import Agent, Player, RL_Agent, TrainedGomokuAgent

# Use a saved checkpoint (e.g. after running gomoku_rl train)
players = (Player(), TrainedGomokuAgent("checkpoint_iter_100.pt"))
```

Checkpoint path is relative to the directory you run `main.py` from (often the project root).

## Run training (deep RL)

From the project root, so that `gomoku_rl` and `src` are on the path:

```bash
pip install torch   # if not already installed
PYTHONPATH=. python -m gomoku_rl.train --iters 500 --games-per-iter 8 --log-every 10 --save-every 100
```

- **Checkpoints** are saved in the current directory as `checkpoint_iter_<N>.pt` (includes `policy_net` state dict and optionally optimizer/step).
- **CLI options:** `--seed`, `--games-per-iter`, `--batch-size`, `--lr`, `--iters`, `--device`, `--save-every`, `--log-every`.

Training loop: collect self-play games → compute returns/advantages (GAE) → PPO update (mini-batches) → log win ratios and turns per game.

## Board and reward convention (aligned)

- **Board:** 0 = empty, 1 = Player 1, 2 = Player 2 (same as `src.board.Board.base` and `gomoku_rl.env.board.GomokuBoard`).
- **Positions:** `(row, col)` in 0..8; same in GUI, repo agents, and gomoku_rl.
- **State tensor for the NN:** shape `(3, 9, 9)` = [P1 stones, P2 stones, empty] (float 0/1).
- **Rewards (training):** Sparse win (+1) / loss (-1) / draw (0) plus small pattern bonuses (e.g. 3-in-a-row, 4-in-a-row, block opponent 4) that decay over training. The repo’s `calculate_reward()` is used in the GUI for display/API only; training uses the wrapper in `gomoku_rl.env.reward_wrapper`.

## Features (gomoku_rl)

| Component | Description |
|-----------|-------------|
| **CNN policy** | 3-layer CNN backbone, actor head (81 actions, legal-move masking), critic head; optional BatchNorm. |
| **predict API** | `policy_net.predict(board_state, legal_actions=None, deterministic=True) → (row, col)`. Accepts (3,9,9) or (9,9) with 0/1/2. |
| **PPO training** | Clipped surrogate, value loss (MSE), entropy bonus, GAE for advantages, mini-batch updates. |
| **Self-play** | Same policy for both players; trajectories (states, actions, rewards, dones, log_probs, values, legal_masks) for training. |
| **Reward wrapper** | Sparse outcome + pattern bonuses with a decay schedule (e.g. 1.0 → 0.2 over 50k steps). |
| **Metrics** | Win ratio P1/P2 and mean±std turns per game over a sliding window (e.g. 100 games). |
| **MCTS (optional)** | Light MCTS using policy prior and value; blend with raw policy for stronger play. |

## Trained agent in the GUI

`TrainedGomokuAgent` in `src.agent` implements the same interface as `RL_Agent`: `command(board, reward) → (row, col)`. It lazy-loads `gomoku_rl` and `torch`, so the GUI runs without them if you only use `Player` and `RL_Agent`. Pass a checkpoint path to use a saved policy; the repo board is converted to the (3,9,9) state via `GomokuBoard.state_from_repo_board(board.base)`.

---

## How to test

### 1. Test the GUI (human vs random agent)

From the project root:

```bash
cd Reinforcement-learning-CW
pip install -r requirements.txt
python main.py
```

A 9×9 board opens. You are red (P1); click a cell to move. Blue (P2) is the random agent and moves automatically. Play until one side has five in a row; the window closes and the winner is printed in the terminal.

### 2. Test the training pipeline (no GUI)

From the project root, with `torch` installed:

```bash
cd Reinforcement-learning-CW
pip install torch
PYTHONPATH=. python -m gomoku_rl.train --iters 2 --games-per-iter 2 --log-every 1
```

You should see one or two log lines (e.g. `Iter 1 | WinP1=... WinP2=... Turns=...`) and no traceback.

**If you see `RuntimeError: Numpy is not available` or a NumPy 2.x / PyTorch warning:** your PyTorch build was compiled for NumPy 1.x. Fix with either:

- `pip install "numpy<2"` (use NumPy 1.x with your current PyTorch), or  
- Create a new env and install a recent PyTorch that supports NumPy 2: e.g. `pip install torch --upgrade` (check [pytorch.org](https://pytorch.org) for your platform).

### 3. Test the trained agent in the GUI

After a short training run that saves a checkpoint:

```bash
PYTHONPATH=. python -m gomoku_rl.train --iters 20 --games-per-iter 4 --save-every 20
```

Then in `main.py` set:

```python
players = (Player(), TrainedGomokuAgent("checkpoint_iter_20.pt"))
```

Run `python main.py` again. P2 should now use the (lightly) trained policy instead of random moves.
