import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from dqn_agent import DQNAgent
from env_advanced import CarbonAwareComputeEnv


st.set_page_config(page_title="CarbonAware Compute", page_icon="C", layout="wide")
st.title("CarbonAware Compute Scheduler")
st.caption("An RL simulation for running flexible workloads with less energy, cost, and carbon.")

with st.expander("What is this simulation?"):
    st.write("The agent decides whether a flexible compute job should run now, wait, run at reduced capacity, or use battery-backed power. It is rewarded for completing work while reducing grid carbon, energy cost, and missed deadlines.")

episode_length = st.sidebar.slider("Simulation steps", 20, 200, 60)
model_path = st.sidebar.text_input("Policy model", "carbon_agent.pt")
mode = st.sidebar.radio("View", ["Policy run", "Manual decisions", "Baseline comparison"])

with st.sidebar.expander("Technical details"):
    st.caption("The dashboard uses a PyTorch DQN policy and a Gymnasium environment. The default model is the policy trained with train_agent.py.")


@st.cache_resource
def load_agent(path, state_dim, action_dim):
    agent = DQNAgent(state_dim, action_dim)
    if not os.path.exists(path):
        return agent, False
    try:
        agent.q.load_state_dict(torch.load(path, map_location="cpu"))
        return agent, True
    except (RuntimeError, OSError):
        return agent, False


def run_policy(length, path):
    env = CarbonAwareComputeEnv(episode_length=length, seed=42)
    agent, loaded = load_agent(path, env.observation_space.shape[0], env.action_space.n)
    state, _ = env.reset()
    rows = []
    done = False
    while not done:
        action = agent.select_action(state, evaluate=loaded)
        state, reward, terminated, truncated, info = env.step(action)
        rows.append({"step": len(rows) + 1, "action": info["action"],
                     "source": info["source"], "reward": reward,
                     "renewable": info["renewable_pct"],
                     "carbon": info["carbon"], "cost": info["cost"],
                     "completed": info["completed"], "deadline": info["deadline"]})
        done = terminated or truncated
    return pd.DataFrame(rows), loaded


def metric_summary(history):
    final = history.iloc[-1]
    return (int(final["completed"]), float(final["carbon"]),
            float(final["cost"]), float(history["reward"].sum()))


if mode == "Policy run":
    history, loaded = run_policy(episode_length, model_path)
    if not loaded:
        st.warning("No compatible carbon_agent.pt found. Run train_agent.py first.")
    completed, carbon, cost, reward = metric_summary(history)
    cards = st.columns(4)
    cards[0].metric("Jobs completed", completed)
    cards[1].metric("Cumulative carbon", f"{carbon:.1f}")
    cards[2].metric("Energy cost", f"{cost:.2f}")
    cards[3].metric("Episode reward", f"{reward:.1f}")

    st.subheader("Scheduling timeline")
    st.caption("Each row is one incoming workload. Reward combines completion, carbon, cost, and deadline pressure.")
    st.dataframe(history[["step", "action", "source", "reward", "deadline"]],
                 use_container_width=True, hide_index=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(history["renewable"], color="#2e8b57", label="renewable availability")
    axes[0].set_ylabel("Renewable %")
    axes[0].legend(loc="upper right")
    axes[1].plot(history["carbon"], color="#b24c34", label="cumulative carbon")
    axes[1].plot(history["cost"], color="#d49a2a", label="cumulative cost")
    axes[1].set_xlabel("Simulation step")
    axes[1].legend(loc="upper left")
    st.pyplot(figure)

elif mode == "Manual decisions":
    st.subheader("Explore the scheduling tradeoffs")
    st.caption("Try deferring a low-priority job, then compare that choice with running immediately or using stored energy.")
    if "manual_env" not in st.session_state or st.button("Reset simulation"):
        st.session_state.manual_env = CarbonAwareComputeEnv(episode_length=episode_length, seed=7)
        st.session_state.manual_state, _ = st.session_state.manual_env.reset()
        st.session_state.manual_rows = []
    env = st.session_state.manual_env
    st.write({"renewable availability": f"{env.renewable_pct:.0%}",
              "battery": f"{env.battery_kwh / env.battery_capacity:.0%}",
              "job demand": f"{env.job['demand']:.0f} kWh",
              "priority": env.job["priority"], "deadline": env.job["deadline"]})
    action = st.selectbox("Decision", list(env.ACTIONS.values()))
    if st.button("Apply decision") and env.steps < episode_length:
        action_id = next(key for key, value in env.ACTIONS.items() if value == action)
        state, reward, _, _, info = env.step(action_id)
        st.session_state.manual_state = state
        st.session_state.manual_rows.append({"step": env.steps, "decision": action,
                                             "executed": info["executed"], "reward": reward})
    if st.session_state.manual_rows:
        st.dataframe(pd.DataFrame(st.session_state.manual_rows),
                     use_container_width=True, hide_index=True)

else:
    st.subheader("Learned policy versus simple strategies")
    history, _ = run_policy(episode_length, model_path)
    strategies = {"RL policy": float(history["reward"].sum())}
    for name, action in {"Always defer": 1, "Always run": 0,
                         "Always reduce": 2}.items():
        env = CarbonAwareComputeEnv(episode_length=episode_length, seed=42)
        state, _ = env.reset()
        total = 0.0
        for _ in range(episode_length):
            state, reward, done, _, _ = env.step(action)
            total += reward
            if done:
                break
        strategies[name] = total
    st.bar_chart(pd.DataFrame.from_dict(strategies, orient="index", columns=["reward"]))
    st.write("Higher reward indicates a better balance between completing work and avoiding carbon, cost, and missed deadlines.")
