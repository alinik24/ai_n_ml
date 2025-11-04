# Autonomous Mobile Robot Navigation with Deep Reinforcement Learning

Multi-algorithm implementation for warehouse robot navigation including Q-Learning, DQN, Double DQN, and Dueling DQN, optimized for NVIDIA RTX 4060.

## Project Overview

This project implements and compares multiple reinforcement learning algorithms for autonomous mobile robot (AMR) navigation in a warehouse environment with:
- Dynamic obstacles (humans, forklifts, other robots)
- Battery management system
- Multi-stage tasks (pickup and delivery)
- Safety constraints and zone authorization

### Implemented Algorithms

**Mandatory (Course Requirements):**
1. **Tabular Q-Learning** - Discretized state space with Q-table
2. **Simple DQN** - Deep Q-Network with single neural network

**Advanced (State-of-the-Art):**
3. **Double DQN** - Decoupled action selection/evaluation (van Hasselt et al., 2016)
4. **Dueling DQN** - Separate value and advantage streams (Wang et al., 2016)

**Baselines (Comparison):**
5. **Random Agent** - Random action selection
6. **Greedy Agent** - Always moves toward goal
7. **Rule-Based Agent** - Hand-coded heuristics

## Installation

### Requirements
- **Python:** 3.10–3.12
- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4060)
- **CUDA:** 12.9
- **PyTorch:** 2.8.0+cu129

### Setup

1. **Create Virtual Environment:**
```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

2. **Install Core Dependencies:**
```powershell
pip install -r requirements.txt
```

3. **Install PyTorch with CUDA 12.9:**
```powershell
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129
```

4. **Verify GPU Installation:**
```powershell
python check_gpu.py
```

Expected output:
```
[OK] CUDA Available: True
[OK] GPU Name: NVIDIA GeForce RTX 4060 Laptop GPU
[OK] CUDA Version: 12.9
```

## Usage

### Option 1: Train All Algorithms (Recommended)

Train all 4 RL algorithms in one run (~2.5-3 hours on RTX 4060):

```powershell
python train_all_algorithms.py
```

This will:
- Train Q-Learning (3000 episodes)
- Train Simple DQN (4000 episodes)
- Train Double DQN (4000 episodes)
- Train Dueling DQN (4000 episodes)
- Generate learning curve comparisons
- Save all models to `models/`

### Option 2: Train Individually

**Train Mandatory Algorithms Only:**
```powershell
python main.py
```

**Train Advanced Algorithms Separately:**
```powershell
python train_advanced.py
```

### Evaluation

Evaluate all trained agents and generate comparison metrics:

```powershell
python src/evaluation/evaluate_all.py
```

This generates:
- `results/metrics/agent_comparison.csv` - Performance table
- `results/figures/agent_comparison.png` - Bar chart comparisons
- `results/figures/radar_comparison.png` - Multi-metric radar chart

### Visualize Training Progress

```powershell
tensorboard --logdir=logs
```

## Project Structure

```
nazarikhah_2163370_ai_n_ml/
├── src/
│   ├── environment/          # Warehouse environment
│   │   ├── __init__.py
│   │   └── warehouse_amr_env.py
│   ├── agents/               # All agent implementations
│   │   ├── __init__.py
│   │   ├── q_learning_agent.py      # Tabular Q-Learning
│   │   ├── dqn_agent.py             # Simple DQN (single network)
│   │   ├── advanced_dqn.py          # Double DQN & Dueling DQN
│   │   └── baseline_agents.py       # Random, Greedy, Rule-Based
│   ├── training/             # Training loops
│   │   ├── __init__.py
│   │   ├── train_qlearning.py
│   │   └── train_dqn.py
│   └── evaluation/           # Evaluation & metrics
│       ├── __init__.py
│       └── evaluate_all.py
├── models/                   # Saved model checkpoints
├── results/
│   ├── figures/             # Plots and visualizations
│   └── metrics/             # CSV performance data
├── main.py                  # Train mandatory algorithms
├── train_all_algorithms.py  # Train all 4 RL algorithms
├── check_gpu.py            # GPU verification script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── STATE_OF_THE_ART_APPROACHES.md  # Technical documentation
└── REFERENCES.md          # Literature citations

```

## GPU Optimizations

All deep learning approaches are optimized for NVIDIA RTX 4060 (8.59 GB VRAM):

- **Mixed Precision Training (FP16/FP32):** 1.5-2x speedup with torch.amp
- **Non-blocking GPU Transfers:** Overlaps data transfer with computation
- **cuDNN Optimizations:** Automatic algorithm selection for convolutions
- **Batch Size Tuning:** Optimized for 8GB VRAM (batch_size=64)

## Expected Results

### Training Time (RTX 4060 with Mixed Precision)
| Algorithm | Episodes | Time | VRAM |
|-----------|----------|------|------|
| Q-Learning | 3,000 | ~10 min | 2 GB RAM |
| Simple DQN | 4,000 | ~45 min | 1.5 GB |
| Double DQN | 4,000 | ~50 min | 2.2 GB |
| Dueling DQN | 4,000 | ~55 min | 2.4 GB |
| **Total** | **15,000** | **~2.5 hrs** | **< 3 GB** |

### Success Rate (Expected)
| Algorithm | Success Rate | Convergence |
|-----------|-------------|-------------|
| Random | 2-5% | Never |
| Greedy | 25-35% | N/A |
| Rule-Based | 45-60% | N/A |
| **Q-Learning** | **95-99%** | ~2500 ep |
| Simple DQN | 80-90% | ~3500 ep |
| **Double DQN** | **92-97%** | ~3000 ep |
| **Dueling DQN** | **90-95%** | ~3200 ep |

## Documentation

- **Technical Details:** See [STATE_OF_THE_ART_APPROACHES.md](STATE_OF_THE_ART_APPROACHES.md)
- **Literature References:** See [REFERENCES.md](REFERENCES.md)
- **Course Requirements:** Project complies with Sapienza ML course mandatory requirements

## Key Features

### Environment
- 20×20 grid warehouse with 12-channel observation (robot, goal, battery, obstacles, etc.)
- Dynamic obstacles with realistic movement patterns
- Battery management (depletes over time, recharge at stations)
- Multi-stage tasks: navigate → pickup → deliver
- Safety rewards (collision penalties, battery depletion)

### Agents
- **Q-Learning:** State discretization with optimistic initialization (Q=10.0)
- **Simple DQN:** Single CNN network (requirement compliant!)
- **Double DQN:** Decoupled action selection/evaluation to reduce overestimation
- **Dueling DQN:** Separate V(s) and A(s,a) streams for better generalization

### Training Features
- Experience replay buffer (capacity: 100,000)
- Epsilon-greedy exploration (ε: 1.0 → 0.01)
- Gradient clipping (max_norm=1.0)
- Early stopping and model checkpointing
- Comprehensive logging and visualization

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in agent initialization:
```python
agent = DQNAgent(batch_size=32)  # Default is 64
```

### ImportError: No module named 'src'
Ensure you're running from project root:
```powershell
cd C:\mydesktop\resproj_thesis\nazarikhah_2163370_ai_n_ml
python main.py
```

### Q-Learning Pickle Error
The project uses regular dict (not defaultdict) for pickle compatibility. This is already fixed in the current version.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nazarikhah2025warehouse,
  title={Autonomous Mobile Robot Navigation with Deep Reinforcement Learning},
  author={Nazarikhah, [First Name]},
  year={2025},
  institution={Sapienza University of Rome}
}
```

## References

See [REFERENCES.md](REFERENCES.md) for complete literature citations.

## License

This project is for academic purposes as part of the Machine Learning course at Sapienza University of Rome.

## Acknowledgments

- Course: Machine Learning, Sapienza University of Rome
- Environment: Custom Gymnasium-based warehouse simulation
- Hardware: NVIDIA RTX 4060 Laptop GPU
- Framework: PyTorch 2.8.0 with CUDA 12.9

---

**Project Status:** ✅ Complete - Ready for evaluation and report writing

**Last Updated:** 2025-11-03
