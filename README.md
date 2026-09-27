# Warehouse AMR Navigation — Reinforcement Learning

Official repository for the Warehouse Autonomous Mobile Robot (AMR) navigation project. The code implements mandatory and optional reinforcement learning agents and a Gym-like warehouse environment used for training and evaluation.

This README is written to represent the repository contents and provide clear instructions for preparing the project to be pushed to GitHub.

---

## Quick summary

- Environment: a Gymnasium-style 20×20 warehouse grid with dynamic obstacles, battery mechanics, pickup/delivery tasks and safety constraints.
- Mandatory agents: Tabular Q-Learning and a Simple DQN (single-network implementation).
- Optional agents: Double DQN, Dueling DQN and several baseline agents for comparison.
- Training: unified interactive script (train.py) for selecting algorithms, device and training length.
- Evaluation: scripts produce publication-quality figures and CSV metrics for analysis.

---

## Repo contents (important files & folders)

Note: push the entire repository to GitHub. Key files and directories are listed below (paths are relative to repository root).

- README.md                                 — this file
- requirements.txt                           — Python dependencies
- train.py                                   — unified interactive training script (recommended entrypoint)

- src/                                       — implementation package
  - __init__.py
  - agents/
    - q_learning_agent.py                    — mandatory tabular Q-Learning agent
    - dqn_agent.py                           — mandatory Simple DQN agent (single network)
    - advanced_dqn.py                        — optional Double/Dueling DQN implementations
    - baseline_agents.py                     — optional baselines (random, greedy, rule-based)
    - __init__.py
  - environment/
    - warehouse_env.py                       — Gym-like AMR warehouse environment
    - __init__.py
  - training/
    - train_qlearning.py                     — training loop for Q-Learning
    - train_dqn.py                           — training loop for DQN
    - train_advanced.py                      — advanced training utilities
    - __init__.py
  - evaluation/
    - evaluate_all.py                        — evaluation and plotting utilities
    - __init__.py

- docs/                                      — additional documentation and report templates
  - report_template.md
  - REPORT.md
  - ADVANCED_ALGORITHMS.md

- models/                                    — (gitignore large binaries if needed) saved checkpoints and model readme
- results/                                   — generated figures and metrics (should be added to .gitignore if large)

- .gitignore                                 — specify files to exclude from Git, e.g. models/ and results/

---

## Installation

1. Create and activate a virtual environment (recommended):

- Windows (PowerShell):
  python -m venv .venv
  .\.venv\Scripts\activate

- Linux / macOS:
  python -m venv .venv
  source .venv/bin/activate

2. Install dependencies:
  pip install -r requirements.txt

3. (Optional) Install a PyTorch build with CUDA if you have an NVIDIA GPU. Example for CUDA 12.9:
  pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129

4. Verify PyTorch / CUDA (optional):
  python -c "import torch; print('cuda:', torch.cuda.is_available())"

---

## How to run

- Train (interactive):
  python train.py

  train.py will:
  - Ask which algorithms to train (Q-Learning and Simple DQN are mandatory)
  - Detect GPU and let you choose device
  - Ask for training length (Quick / Standard / Long)
  - Save checkpoints to models/ and figures to results/figures/

- Evaluate (generate figures / CSV metrics):
  python src/evaluation/evaluate_all.py


---

## Notes on reproducibility and datasets

- The environment is deterministic under seeded runs but contains stochastic actors (humans, forklifts). For reproducible experiments, set seeds where applicable in training scripts.
- Save random seeds and configuration used for training alongside models and metrics (a good place is models/<model_name>_meta.json).

---

## Contributing

Contributions are welcome. Suggested workflow:
- Fork the repository
- Create a feature branch
- Add tests and documentation for new features
- Submit a pull request with a clear description

---

## License

This repository is provided for academic purposes. Add a LICENSE file if you wish to make the project open-source under a specific license.

---

If you want, I can also:
- Create/update a concise models/README.md describing checkpoint files
- Add a minimal .github/workflows/CI configuration (GitHub Actions) to run linting/tests on push
- Produce a cleaned file list (tree) to include in the repo root

Tell me which of the above you'd like me to add or modify before you push to GitHub.
