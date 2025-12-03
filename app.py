import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from collections import defaultdict

# Try importing env and agent from common filenames
try:
    from env_advanced import GeothermalEnvAdvanced as GeothermalEnv
except Exception:
    try:
        from env import GeothermalEnv as GeothermalEnv
    except Exception:
        GeothermalEnv = None

try:
    from dqn_agent import DQNAgent
except Exception:
    try:
        from dqn import DQNAgent
    except Exception:
        DQNAgent = None

st.set_page_config(layout="wide", page_title="GeoCompute Dashboard")

st.title("GeoCompute — Geothermal Scheduler (Interactive Demo)")

# Sidebar: model & mode selection
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model path", "geo_agent.pt")
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

mode = st.sidebar.selectbox("Mode", ["Run Greedy Episode", "Step-through Manual", "Compare Heuristics"])
episode_len = st.sidebar.number_input("Episode length", min_value=10, max_value=500, value=60)

load_model_btn = st.sidebar.button("Load model")

@st.cache_resource
def load_agent_safe(model_path, state_dim, action_dim, device="cpu"):
    if DQNAgent is None:
        st.error("DQNAgent class not found. Put your agent class in dqn_agent.py or dqn.py")
        return None
    agent = DQNAgent(state_dim, action_dim, device=device)
    try:
        sd = torch.load(model_path, map_location=device)
        agent.q.load_state_dict(sd)
        st.success(f"Loaded model from {model_path}")
    except Exception as e:
        st.warning(f"Failed to load model: {e}. Agent created untrained.")
    return agent

# show quick status
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Environment")
    if GeothermalEnv is None:
        st.error("Environment class not found. Place GeothermalEnvAdvanced in env_advanced.py")
    else:
        st.success("Geothermal environment class found")
with col2:
    st.markdown("### Agent")
    if DQNAgent is None:
        st.error("DQNAgent not found. Place DQNAgent in dqn_agent.py")
    else:
        st.success("DQNAgent class found")

# instantiate env for UI controls
if GeothermalEnv:
    env = GeothermalEnv(episode_length=int(episode_len))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
else:
    env = None
    state_dim = None
    action_dim = None

# Agent loading
agent = None
if load_model_btn and env is not None:
    agent = load_agent_safe(model_path, state_dim, action_dim, device=device)
else:
    # allow lazy loading
    if env is not None and DQNAgent is not None:
        agent = DQNAgent(state_dim, action_dim, device=device)

# Helper: run one (greedy) episode and return history
def run_episode_greedy(env, agent, episode_length):
    history = defaultdict(list)
    s, _ = env.reset()
    done = False
    step = 0
    while not done and step < episode_length:
        # compute valid actions for display (but agent may ignore)
        valid_actions = []
        # try to infer env internals safely
        try:
            if hasattr(env, "job") and hasattr(env, "reservoir_heat"):
                if env.job <= env.reservoir_heat and (not hasattr(env, "max_rate") or env.job <= env.max_rate):
                    valid_actions.append(0)
            if hasattr(env, "battery_kwh") and env.job <= env.battery_kwh:
                valid_actions.append(1)
        except Exception:
            valid_actions = list(range(env.action_space.n))

        a = agent.select_action(s, valid_actions, ) if agent else np.random.randint(env.action_space.n)
        ns, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # store
        history["step"].append(step)
        history["state"].append(s)
        history["action"].append(a)
        history["source"].append(info.get("source", None))
        history["reward"].append(r)
        # try to get readable env internals
        history["job"].append(getattr(env, "job", None))
        history["reservoir"].append(getattr(env, "reservoir_heat", None))
        history["battery"].append(getattr(env, "battery_kwh", None))
        history["priority"].append(info.get("priority", None))
        s = ns
        step += 1
    return history

# UI: run controls
if mode == "Run Greedy Episode":
    st.header("Greedy Episode (agent policy)")
    colA, colB = st.columns([1,2])
    with colA:
        if st.button("Run Episode (Greedy)"):
            if env is None:
                st.error("Environment not loaded.")
            else:
                if agent is None:
                    st.warning("Agent uninitialized — will run random actions.")
                hist = run_episode_greedy(env, agent, episode_len)
                # plots
                fig, axes = plt.subplots(3,1, figsize=(10,8), sharex=True)
                axes[0].plot(hist["reservoir"]); axes[0].set_ylabel("Reservoir heat")
                axes[1].plot(hist["battery"]); axes[1].set_ylabel("Battery kWh")
                axes[2].plot(hist["reward"]); axes[2].set_ylabel("Reward")
                st.pyplot(fig)
                # action table
                import pandas as pd
                df = pd.DataFrame({
                    "step": hist["step"],
                    "action": hist["action"],
                    "source": hist["source"],
                    "job_next": hist["job"],
                    "priority": hist["priority"],
                    "reward": hist["reward"]
                })
                st.dataframe(df)

if mode == "Step-through Manual":
    st.header("Step-through (Manual control)")
    if env is None:
        st.error("No environment available.")
    else:
        if "manual_state" not in st.session_state or st.button("Reset manual episode"):
            st.session_state.manual_env = env.__class__(episode_length=episode_len)
            st.session_state.manual_state, _ = st.session_state.manual_env.reset()
            st.session_state.history = defaultdict(list)
            st.success("Manual episode reset")
        manual_env = st.session_state.manual_env
        s = st.session_state.manual_state

        st.subheader("Current observation")
        st.write(s)

        st.subheader("Choose action")
        a = st.selectbox("Action (0=Geo,1=Battery,2=Grid)", [0,1,2])
        if st.button("Step"):
            ns, r, done, tr, info = manual_env.step(a)
            st.session_state.history["action"].append(a)
            st.session_state.history["info"].append(info)
            st.session_state.history["reward"].append(r)
            st.session_state.history["reservoir"].append(getattr(manual_env, "reservoir_heat", None))
            st.session_state.history["battery"].append(getattr(manual_env, "battery_kwh", None))
            st.session_state.manual_state = ns
            st.write("Step executed:", info, "reward:", r)
            if done:
                st.success("Episode ended")

        if len(st.session_state.history.get("action", [])) > 0:
            fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True)
            ax[0].plot(st.session_state.history["reservoir"]); ax[0].set_ylabel("Reservoir")
            ax[1].plot(st.session_state.history["battery"]); ax[1].set_ylabel("Battery")
            st.pyplot(fig)
            st.table({
                "step": list(range(len(st.session_state.history["action"]))),
                "action": st.session_state.history["action"],
                "info": st.session_state.history["info"],
                "reward": st.session_state.history["reward"]
            })

if mode == "Compare Heuristics":
    st.header("Compare Heuristics vs Agent (single episode)")
    heur = st.selectbox("Heuristic", ["greedy_geo_then_batt", "always_grid", "geo_if_possible_else_grid"])
    if st.button("Run comparison"):
        # run agent episode
        agent_hist = run_episode_greedy(env, agent, episode_len) if agent else None

        # heuristic 1: geo if possible, else battery if possible, else grid
        def run_heuristic(env_cls, heuristic):
            env2 = env_cls(episode_length=episode_len)
            s,_ = env2.reset()
            done=False
            history = defaultdict(list)
            while not done:
                job = env2.job
                # decide
                if heuristic == "greedy_geo_then_batt":
                    if job <= getattr(env2, "reservoir_heat", 0) and (not hasattr(env2, "max_rate") or job <= env2.max_rate):
                        a = 0
                    elif job <= getattr(env2, "battery_kwh", 0):
                        a = 1
                    else:
                        a = 2
                elif heuristic == "always_grid":
                    a = 2
                else:
                    a = 0 if job <= getattr(env2, "reservoir_heat", 0) else 2
                ns, r, terminated, tr, info = env2.step(a)
                history["reward"].append(r)
                history["action"].append(a)
                history["reservoir"].append(getattr(env2, "reservoir_heat", None))
                history["battery"].append(getattr(env2, "battery_kwh", None))
                done = terminated or tr
            return history

        h_hist = run_heuristic(GeothermalEnv, heur)

        # plot rewards
        fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
        if agent_hist:
            ax[0].plot(agent_hist["reward"], label="agent")
        ax[0].plot(h_hist["reward"], label="heuristic")
        ax[0].legend(); ax[0].set_ylabel("Reward")
        ax[1].plot(h_hist["reservoir"], label="heuristic reservoir")
        if agent_hist:
            ax[1].plot(agent_hist["reservoir"], label="agent reservoir")
        ax[1].legend(); ax[1].set_ylabel("Reservoir / Battery")
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("Model file should be in this folder or specify a path. Use `streamlit run app.py` to run this UI.")
