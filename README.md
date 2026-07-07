# VEC-DRL-Lab

**Vehicular Edge Computing Simulator with Deep Reinforcement Learning Scheduling**

A research simulator built from scratch for studying task offloading policies
in VEC environments. Accompanies the lab manual:
*VEC Research Laboratory — Module 1*.

## Author
**Dr. Habte Lejebo**  
Research Areas: Vehicular Edge Computing · Fog Computing · Deep Reinforcement Learning

## Key Results

| Policy | Completion Rate |
|---|---|
| Always Offload | 41.3% |
| Always Local | 20.0% |
| **DQN Agent** | **59.0%** |

DQN outperforms the best baseline by **17.7 percentage points** through
adaptive balancing of local and edge resources (67.9% edge / 32.1% local).

## Project Structure

VEC-DRL-Lab/
├── src/vecsim/          # Core simulator package
│   ├── task.py          # Task lifecycle model
│   ├── vehicle.py       # Vehicle mobility + local execution
│   ├── edge_server.py   # RSU processing + queuing + handover
│   ├── channel.py       # Shannon wireless channel model
│   └── agent.py         # DQN scheduling agent
├── experiments/
│   ├── simulation.py    # 1000-episode DRL training loop
│   ├── baseline_comparison.py  # Policy evaluation
│   └── plot.py          # Learning curve visualisation
├── results/             # Metrics, plots, trained weights
└── docs/                # Lab manual PDF