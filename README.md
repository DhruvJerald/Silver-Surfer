**Silver Surfer**

Godot + Reinforcement Learning + Rule-Based + Hybrid AI

A fast paced 2D endless runner built in Godot,combined with multiple AI paradigms to explore the learning efficiency, behavior emergence, and decision optimization of the models and the rule based bot


-----> https://dhruvj.itch.io/silver-surfer-base-game <-----(GAME LINK **HIGHLY RECCOMEND DOWNLOADING THE ACTUAL GAME AS THE WEB SOCKET CAPS THE QUALITY AND OVERALL GAME EXPERIENCE**)



Project Overview(my goal)

Silver Surfer is not just a game it was my first experiment with the technical aspects of building an AI model

The goal of this project is to explore:

Which model or technique can secure me 1000+ points with ease destroying the human competition

**Core**

A simple 2D endless runner:

Player moves between 3 lanes(0,1,2)
Controls:
W → Move Up
S → Move Down

**"ESC"--to pause the game**  

Objective:
Avoid obstacles
Survive as long as possible
Maximize score

 **Game X AI Framework**

1)  Godot Game Engine (Environment)   
          ⇅ TCP Socket
    Python RL Server (Agent)

2)  Godot Game Engine (Environment)
          ⇅ TCP Socket
   Python Hybrid (Model)

3)  Godot Game engine(Environment + Rule Based Bot)

this is the data flow b/w tcp connection:-->

Godot sends:
Current state (distances, lane, speed)
Reward signal
Done flag
Python agent:
Processes state
Chooses action
Sends action back
Godot:
Applies action
Continues simulation
State Representation

The agent receives:(JSON PACKETS)

[
  lane,        # normalized (0–2)
  t1, m1, b1,  # closest obstacle distances
  t2, m2, b2,  # second obstacle distances
  speed        # normalized
]

-->Systems Implemented


1️ Rule-Based Agent

A bot using simple logic:

Avoid closest obstacle
Prioritize safest lane
React based on thresholds

✔ Fast
❌ Not adaptive
❌ Limited ceiling

2️ Reinforcement Learning Agent (DQN)

Deep Q-Network using:

PyTorch
Experience Replay
Prioritized Sampling
Target Network Stabilization
Key Features:
Epsilon-Greedy exploration
Smooth L1 Loss (Huber Loss)
Soft target updates
Action repeat for stability
Q(s, a) → expected future reward
CSV logging to study and understand the models learning patterns
Safe quit from terminal to save the model

3️  Hybrid Learning Agent 

(A sum from my previous mistakes) 

Rule-Based decision system
Reinforcement Learning policy
How it works:
RL proposes an action
Rule system validates / overrides in critical situations
Blended decision improves survival
Why this matters:
Fixes early-stage RL stupidity
Prevents catastrophic failures
Accelerates learning convergence
Produces more human-like behavior
CSV logging to study and understand the models learning patterns
Safe quit from terminal to save the model

✔ Stable
✔ Smarter early training
✔ Better long-term performance

**Reward System**

Designed to guide learning(just enough to not count as rules)

✅ Survival reward (small positive)
✅ Passing obstacle bonus (main signal)
⚠️ Danger penalty (too close to obstacle)
⚠️ Movement penalty (anti-spam)
❌ Death penalty (large negative)
    Experiment Goals

