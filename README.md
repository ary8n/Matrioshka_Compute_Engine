# CarbonAware Compute Scheduler

CarbonAware Compute is a reinforcement-learning simulation for scheduling flexible computing workloads under energy, cost, capacity, and carbon constraints.

The agent observes renewable-energy availability, battery charge, available server capacity, grid carbon intensity, electricity price, workload demand, job priority, and deadline. It then chooses whether to run the job immediately, defer it, run it at reduced capacity, or use battery-backed power.

The goal is to complete useful work while reducing grid emissions and operating cost without missing urgent deadlines.

## Demo

Run the Streamlit dashboard locally:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes three views:

- **Policy run:** watch the trained DQN schedule a complete episode.
- **Manual decisions:** explore the tradeoffs behind individual scheduling choices.
- **Baseline comparison:** compare the learned policy with simple fixed strategies.

## How it works

Each simulation step presents the agent with a new workload. The environment updates server load, battery storage, renewable availability, carbon intensity, cost, and deadline pressure after the decision. Rewards favor completed work and penalize carbon emissions, energy cost, invalid actions, and missed deadlines.

The optional policy is implemented with PyTorch and trained in `train_agent.py`. The hosted dashboard does not require PyTorch: it uses a transparent carbon-aware fallback policy when the training stack is unavailable. Install `requirements-training.txt` locally to train or evaluate the DQN model.

## Portfolio framing

This is a research prototype, not a production data-center controller. It demonstrates how reinforcement learning can model carbon-aware workload scheduling and make the tradeoffs visible through an interactive simulation.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit visualization and interactive demo |
| `env_advanced.py` | Carbon-aware Gymnasium environment |
| `dqn_agent.py` | Deep Q-network agent |
| `train_agent.py` | Training script for the policy |
| `carbon_agent.pt` | Trained policy weights |

## Re-train the policy

```powershell
python train_agent.py --episodes 500 --episode-length 60 --output carbon_agent.pt
```