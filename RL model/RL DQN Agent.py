import socket
import json
import random
import numpy as np
import csv
import os
from collections import deque
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim


# ===========================================
# CONFIG (defining all the hyperparameters)
# ===========================================
STATE_SIZE = 8
ACTION_SIZE = 3

LR = 0.0005
GAMMA = 0.99

BATCH_SIZE = 128
MEMORY_SIZE = 30000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 500

MODEL_FILE = "dqn_model.pth"
CSV_FILE = "training_log.csv"

ACTION_REPEAT = 4
last_action = 0
repeat_count = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# =============================
# NETWORK
# =============================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# =============================
# SAFE LOAD
# =============================
if os.path.exists(MODEL_FILE):
    try:
        policy_net.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        print("💾 Model loaded")
    except:
        print("⚠️ Model mismatch, starting fresh")


def save_model():
    torch.save(policy_net.state_dict(), MODEL_FILE)
    print("💾 Model saved")

def handle_exit(sig, frame):
    save_model()
    csv_file.close()
    print("👋 Exiting safely")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
# =============================
# MEMORY + PER
# =============================
memory = []
priorities = []

def store_transition(s, a, r, s2, done):
    max_prio = max(priorities) if priorities else 1.0

    memory.append((s, a, r, s2, done))
    priorities.append(max_prio)

    if len(memory) > MEMORY_SIZE:
        memory.pop(0)
        priorities.pop(0)

def sample_batch():
    probs = np.array(priorities) ** 0.6
    probs /= probs.sum()

    indices = np.random.choice(len(memory), BATCH_SIZE, p=probs)
    batch = [memory[i] for i in indices]

    s, a, r, s2, d = zip(*batch)

    return (
        torch.tensor(np.array(s), dtype=torch.float32).to(device),
        torch.tensor(a).to(device),
        torch.tensor(r, dtype=torch.float32).to(device),
        torch.tensor(np.array(s2), dtype=torch.float32).to(device),
        torch.tensor(d, dtype=torch.float32).to(device),
        indices
    )

# =============================
# STATE
# =============================
def process_state(s):
    def norm(x): return min(x / 300.0, 1.0)

    return np.array([
        s["lane"] / 2.0,
        norm(s["t1"]), norm(s["m1"]), norm(s["b1"]),
        norm(s["t2"]), norm(s["m2"]), norm(s["b2"]),
        s["speed"] / 1000.0
    ], dtype=np.float32)

def get_epsilon(ep):
    return max(EPS_END, EPS_START * np.exp(-ep / EPS_DECAY))
# =============================
# ACTION
# =============================
def choose_action(state, epsilon):
    # 🔥 FORCE RANDOM BURSTS
    if random.random() < max(epsilon, 0.1):
        return random.randint(0, ACTION_SIZE - 1)

    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(s)
        return int(torch.argmax(q_values).item())
    
# 🔥 Anti-stuck: random override sometimes
if random.random() < 0.05:
    action = random.randint(0, ACTION_SIZE - 1)

# =============================
# TRAIN
# =============================
train_step = 0

def train():
    global train_step

    if len(memory) < 300:
        print("⏳ Not enough memory:", len(memory))
        return

    s, a, r, s2, d, indices = sample_batch()

    a = a.long()

    q = policy_net(s)
    current_q = q.gather(1, a.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_actions = policy_net(s2).argmax(1)
        next_q = target_net(s2).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q = r + GAMMA * next_q * (1 - d)

    loss = nn.SmoothL1Loss()(current_q, target_q)

    # update priorities
    td_error = torch.abs(current_q - target_q).detach().cpu().numpy()
    for i, idx in enumerate(indices):
        priorities[idx] = float(td_error[i]) + 1e-5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    train_step += 1

    # soft update
    if train_step % 4 == 0:
        tau = 0.01
        for t, p in zip(target_net.parameters(), policy_net.parameters()):
            t.data.copy_(tau * p.data + (1 - tau) * t.data)
# =============================
# CSV
# =============================
csv_file = open(CSV_FILE, "a", newline="")
csv_writer = csv.writer(csv_file)
if csv_file.tell() == 0:
    csv_writer.writerow(["episode", "score", "avg10", "best", "epsilon"])

# =============================
# SERVER
# =============================
def process_buffer(buffer):
    lines = []
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        if line.strip():
            lines.append(line.strip())
    return lines, buffer

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 5555))
server.listen(1)

print("🔌 Waiting for Godot...")

episode_rewards = []
episode_count = 0
best_score = 0

while True:
    conn, addr = server.accept()
    print("✅ Connected")

    buffer = ""
    prev_state = None
    prev_action = None
    current_reward = 0.0

    repeat_count = 0
    last_action_local = 0

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            buffer += data.decode()
            lines, buffer = process_buffer(buffer)

            for line in lines:
                try:
                    packet = json.loads(line)
                except Exception as e:
                    print("⚠️ JSON parse error:", e)
                    continue

                state_raw = packet.get("state", {})
                reward = float(packet.get("reward", 0.0))
                done = bool(packet.get("done", False))

                # -------------------
                # PROCESS STATE
                # -------------------
                state = process_state(state_raw)
                epsilon = get_epsilon(episode_count)

                # -------------------
                # STORE TRANSITION
                # -------------------
                if prev_state is not None:
                    store_transition(prev_state, prev_action, reward, state, done)

                # -------------------
                # ACTION SELECTION
                # -------------------
                if repeat_count == 0:
                    new_action = choose_action(state, epsilon)
                    last_action_local = new_action




                action = last_action_local
                repeat_count = (repeat_count + 1) % ACTION_REPEAT

                # -------------------
                # SEND TO GODOT
                # -------------------
                try:
                    packet_out = {"action": int(action), "seed": 42}
                    conn.sendall((json.dumps(packet_out) + "\n").encode())
                except Exception as e:
                    print("⚠️ Send failed:", e)
                    break

                # -------------------
                # TRAIN
                # -------------------
                train()

                prev_state = state
                prev_action = action
                current_reward += reward

                if done:
                    break

    except Exception as e:
        print("⚠️ Error:", e)

    finally:
        episode_rewards.append(current_reward)
        episode_count += 1

        if episode_count % 10 == 0:
            torch.save(policy_net.state_dict(), MODEL_FILE)
            print("💾 Model saved")

        best_score = max(best_score, current_reward)

        avg10 = sum(episode_rewards[-10:]) / max(len(episode_rewards[-10:]), 1)
        epsilon = max(0.2, get_epsilon(episode_count))

        print(f"📊 Ep {episode_count} | Score {current_reward:.0f} | Avg10 {avg10:.1f} | Best {best_score:.0f}")

        csv_writer.writerow([episode_count, current_reward, avg10, best_score, epsilon])
        csv_file.flush()

        save_model()

        conn.close()
        print("🔌 Disconnected. Waiting...") 