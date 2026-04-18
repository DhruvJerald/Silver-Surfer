Silver Surfer

Godot + Reinforcement Learning + Rule-Based + Hybrid AI

A fast paced 2D endless runner built in Godot, combined with multiple AI paradigms to explore learning efficiency, behavior emergence, and decision-making strategies.(thats what AI had to say ide like to call it a game with AI)

Play the Game

Download the full version here:
https://dhruvj.itch.io/silver-surfer-base-game

Note: The downloadable version is recommended. The browser version is limited and does not support the full AI pipeline.

Project Overview

Silver Surfer started as an experiment to understand how different AI approaches perform in the same controlled environment.

The core question behind this project:

Which system can consistently reach high scores (1000+) and outperform human gameplay?

This project compares:

Rule-based logic
Reinforcement learning
A hybrid combination of both
Game Mechanics

A simple 2D endless runner:

The player moves between 3 lanes: 0 (top), 1 (middle), 2 (bottom)
Controls
W → Move up
S → Move down
ESC → Pause
Objective
Avoid obstacles
Survive as long as possible
Maximize score
System Architecture

The project is built around a Godot ↔ Python communication loop using TCP sockets.

Setup Variants
1. Reinforcement Learning Agent
Godot (Environment)
        ⇅ TCP
Python (DQN Agent)
2. Hybrid Agent
Godot (Environment)
        ⇅ TCP
Python (Hybrid Logic + RL)
3. Rule-Based Agent
Godot (Environment + Bot Logic)
Data Flow

Each frame:

Godot sends:

Current state (lane, obstacle distances, speed)
Reward
Done flag

Python agent:

Processes state
Selects action

Godot:

Applies action
Continues simulation
State Representation

The agent receives a JSON packet:

[
  lane,
  t1, m1, b1,
  t2, m2, b2,
  speed
]

Where:

t1, m1, b1 → nearest obstacle distances
t2, m2, b2 → second nearest distances
speed → normalized game speed
Implemented Systems
1. Rule-Based Agent

Simple deterministic logic:

Avoid nearest obstacle
Move to safest lane
Use threshold-based decisions

Pros

Fast
Predictable

Cons

Not adaptive
Limited performance ceiling
2. Reinforcement Learning Agent (DQN)

Built using:

PyTorch
Experience Replay
Prioritized Sampling
Target Network

Key Features

Epsilon-greedy exploration
Huber (Smooth L1) loss
Soft target updates
Action repeat for stability
CSV logging for training analysis

Core Idea

Q(s, a) → expected future reward
3. Hybrid Learning Agent

Combines:

Rule-based safety layer
Reinforcement learning policy

How it works

RL proposes an action
Rule system overrides unsafe decisions
Final action is executed

Why this matters

Prevents early-stage RL failures
Stabilizes training
Improves convergence speed
Produces more consistent behavior

Characteristics

More stable than pure RL
Better early performance
Strong long-term results
Reward System

Designed to guide learning without hardcoding behavior:

Small survival reward
Large reward for passing obstacles
Penalty for being too close to obstacles
Small penalty for unnecessary movement
Large penalty for death

**Installation and Setup**
Requirements
Godot Engine (4.x recommended)
Python 3.9+
PyTorch
NumPy
1. Clone the Repository
git clone https://github.com/your-username/silver-surfer.git
cd silver-surfer
2. Install Python Dependencies
pip install torch numpy
3. Run the AI Server
python main.py
4. Run the Game
Open the project in Godot
Run the main scene

The game will connect to the Python agent via TCP.

Reproducing Results

To replicate experiments:

Start Python training server(PYTHON BEFORE GODOT)
Run the Godot game
Let episodes run continuously
Monitor:
Terminal logs
CSV training file
Stop safely using CTRL + C (model will save)
