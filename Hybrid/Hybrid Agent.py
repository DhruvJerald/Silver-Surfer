import socket, json, random, numpy as np, os
from collections import deque
import time
import signal
import sys
import csv
from datetime import datetime
import threading

# =============================
# CONFIG
# =============================
BEST_RUN_FILE = "best_run.json"
TRAINING_LOG_FILE = "training_log.csv"
CHECKPOINT_FILE = "checkpoint.json"
EXPLORE_WINDOW = 15          # Try different actions only within this many frames before death
LANE_CHANGE_COOLDOWN = 3     # Frames to wait after changing lanes
MIN_IMPROVEMENT = 1          # Minimum score increase to save new best

# Manual override
MANUAL_MODE = None  # 'replay', 'explore', or None for auto
manual_save_request = False
should_exit = False

# =============================
# UTILITIES
# =============================
def to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [to_native(x) for x in obj]
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    return obj

def save_best_run(actions, score, episode, death_frame):
    data = {
        "actions": [int(a) for a in actions],
        "score": float(score),
        "episode": int(episode),
        "death_frame": int(death_frame)
    }
    with open(BEST_RUN_FILE, "w") as f:
        json.dump(data, f)

def load_best_run():
    if os.path.exists(BEST_RUN_FILE):
        try:
            with open(BEST_RUN_FILE, "r") as f:
                data = json.load(f)
                return data.get("actions", []), data.get("score", 0), data.get("death_frame", 0)
        except: pass
    return [], 0, 0

def save_checkpoint(episode, best_actions, best_score, death_frame):
    data = {
        "episode": int(episode),
        "best_actions": [int(a) for a in best_actions],
        "best_score": float(best_score),
        "death_frame": int(death_frame)
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
                return data.get("episode", 0), data.get("best_actions", []), data.get("best_score", 0), data.get("death_frame", 0)
        except: pass
    return 0, [], 0, 0

# =============================
# CSV LOGGING
# =============================
csv_file = None
csv_writer = None

def init_csv():
    global csv_file, csv_writer
    file_exists = os.path.exists(TRAINING_LOG_FILE)
    csv_file = open(TRAINING_LOG_FILE, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['episode', 'score', 'best_score', 'death_frame', 'explore_frame', 'new_action', 'duration', 'timestamp'])

def log_episode(episode, score, best_score, death_frame, explore_frame, new_action, duration):
    if csv_writer:
        csv_writer.writerow([episode, score, best_score, death_frame, explore_frame, new_action, round(duration,2), datetime.now().isoformat()])
        csv_file.flush()

def close_csv():
    global csv_file
    if csv_file:
        csv_file.close()

# =============================
# MANUAL CONTROL THREAD
# =============================
def control_thread():
    global MANUAL_MODE, manual_save_request, should_exit
    print("\n" + "="*60)
    print("CONTROLS:  r = REPLAY mode (exact best run)")
    print("           e = EXPLORE mode (learn new)")
    print("           a = AUTO mode (hybrid)")
    print("           s = SAVE current run as best")
    print("           q = QUIT")
    print("="*60 + "\n")
    while not should_exit:
        try:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 'r':
                MANUAL_MODE = 'replay'
                print("🎮 MANUAL: REPLAY mode (exact best run)")
            elif cmd == 'e':
                MANUAL_MODE = 'explore'
                print("🔍 MANUAL: EXPLORE mode")
            elif cmd == 'a':
                MANUAL_MODE = None
                print("🤖 AUTO mode (hybrid)")
            elif cmd == 's':
                manual_save_request = True
                print("💾 Will save current run after episode")
            elif cmd == 'q':
                should_exit = True
                print("🛑 Quitting...")
                break
        except: pass

# =============================
# RULE-BASED BOT (Heuristic with lane discipline)
# =============================
class RuleBasedBot:
    def __init__(self):
        self.cooldown = 0
        
    def get_action(self, state, speed):
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0
            
        lane = int(state["lane"])
        # distances in pixels
        top = float(state.get("top1", 9999))
        mid = float(state.get("mid1", 9999))
        bot = float(state.get("bot1", 9999))
        
        # time to impact
        t_top = top / speed if speed > 0 else 9999
        t_mid = mid / speed if speed > 0 else 9999
        t_bot = bot / speed if speed > 0 else 9999
        
        reaction = 0.25 * (300 / max(speed, 300))  # faster speed = less reaction time
        
        # danger detection
        danger = [t_top < reaction, t_mid < reaction, t_bot < reaction]
        
        if danger[lane]:
            # current lane dangerous -> find safest lane
            times = [t_top, t_mid, t_bot]
            safest = np.argmax(times)
            if safest < lane:
                self.cooldown = LANE_CHANGE_COOLDOWN
                return 1
            elif safest > lane:
                self.cooldown = LANE_CHANGE_COOLDOWN
                return 2
        return 0

# =============================
# HYBRID AGENT (Replay + Rule + Exploration)
# =============================
class HybridAgent:
    def __init__(self):
        self.best_actions = []
        self.best_score = 0
        self.death_frame = 0
        self.rule_bot = RuleBasedBot()
        self.current_frame = 0
        self.explore_start = -1   # frame where we start exploring
        self.new_action = None
        
    def set_best_run(self, actions, score, death_frame):
        self.best_actions = actions.copy()
        self.best_score = score
        self.death_frame = death_frame
        print(f"\n🏆 Best run loaded: score {score:.0f}, length {len(actions)}, died at frame {death_frame}")
        
    def reset_episode(self):
        self.current_frame = 0
        self.explore_start = -1
        self.new_action = None
        
    def get_action(self, state, speed, current_score, manual_mode):
        # Manual override
        if manual_mode == 'explore':
            # Pure exploration: rule-based + random
            if np.random.random() < 0.3:
                action = np.random.choice([0,1,2])
            else:
                action = self.rule_bot.get_action(state, speed)
            self.current_frame += 1
            return action
            
        if manual_mode == 'replay':
            # Exact replay of best run (no variation)
            if self.current_frame < len(self.best_actions):
                action = self.best_actions[self.current_frame]
                self.current_frame += 1
                return action
            else:
                # Past best run length, use rule-based
                action = self.rule_bot.get_action(state, speed)
                self.current_frame += 1
                return action
        
        # AUTO MODE: Hybrid strategy
        # If no best run yet, use rule-based + exploration
        if not self.best_actions:
            action = self.rule_bot.get_action(state, speed)
            if np.random.random() < 0.2:
                action = np.random.choice([0,1,2])
            self.current_frame += 1
            return action
        
        # We have a best run
        # Determine explore window: last EXPLORE_WINDOW frames before death
        explore_window_start = max(0, self.death_frame - EXPLORE_WINDOW)
        
        if self.current_frame < explore_window_start:
            # Before explore window: EXACTLY follow best run
            if self.current_frame < len(self.best_actions):
                action = self.best_actions[self.current_frame]
                self.current_frame += 1
                return action
            else:
                action = self.rule_bot.get_action(state, speed)
                self.current_frame += 1
                return action
        else:
            # Inside explore window: try different action ONCE
            if not self.new_action and self.current_frame < len(self.best_actions):
                # First time in explore window: choose different action than best run's action at this frame
                best_action = self.best_actions[self.current_frame] if self.current_frame < len(self.best_actions) else 0
                possible = [0,1,2]
                if best_action in possible:
                    possible.remove(best_action)
                self.new_action = np.random.choice(possible) if possible else 0
                print(f"   🔄 Exploring at frame {self.current_frame}: trying {self.new_action} instead of {best_action}")
                action = self.new_action
            elif self.new_action:
                # After exploring, continue with rule-based (since we're in new territory)
                action = self.rule_bot.get_action(state, speed)
            else:
                # Past best run length
                action = self.rule_bot.get_action(state, speed)
            
            self.current_frame += 1
            return action

# =============================
# MAIN TRAINING LOOP
# =============================
def main():
    global MANUAL_MODE, manual_save_request, should_exit
    
    # Load previous data
    best_actions, best_score, best_death_frame = load_best_run()
    start_episode, checkpoint_actions, checkpoint_score, checkpoint_death = load_checkpoint()
    if checkpoint_actions and checkpoint_score > best_score:
        best_actions, best_score, best_death_frame = checkpoint_actions, checkpoint_score, checkpoint_death
        start_episode = max(start_episode, 0)
    
    # Setup
    init_csv()
    signal.signal(signal.SIGINT, lambda s,f: setattr(sys.modules[__name__], 'should_exit', True))
    threading.Thread(target=control_thread, daemon=True).start()
    
    agent = HybridAgent()
    agent.set_best_run(best_actions, best_score, best_death_frame)
    
    # Socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 5555))
    server.listen(1)
    server.settimeout(1.0)
    
    print("\n🚀 Waiting for Godot...")
    episode = start_episode
    
    while episode < 1000 and not should_exit:
        try:
            conn, addr = server.accept()
            conn.settimeout(0.5)
            
            buffer = ""
            current_actions = []
            current_score = 0.0
            frame = 0
            agent.reset_episode()
            
            while True:
                try:
                    data = conn.recv(4096)
                    if not data: break
                except socket.timeout: continue
                except: break
                
                buffer += data.decode()
                lines = buffer.split("\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    if not line.strip(): continue
                    try:
                        packet = json.loads(line)
                    except: continue
                    
                    state = packet.get("state", {})
                    reward = float(packet.get("reward", 0))
                    done = packet.get("done", False)
                    speed = float(state.get("speed", 300))
                    
                    # Get action based on current manual mode
                    action = agent.get_action(state, speed, current_score, MANUAL_MODE)
                    current_actions.append(int(action))
                    
                    # Send action
                    conn.sendall((json.dumps({"action": int(action), "seed": 42}) + "\n").encode())
                    
                    current_score += reward
                    frame += 1
                    
                    if done:
                        death_frame = frame
                        break
                if done:
                    break
            
            episode_duration = time.time() - episode_start if 'episode_start' in dir() else 0
            episode_start = time.time()
            
            # Check for new best score
            is_new_best = False
            if current_score > best_score:
                best_score = current_score
                best_actions = current_actions.copy()
                best_death_frame = death_frame
                is_new_best = True
                save_best_run(best_actions, best_score, episode, best_death_frame)
                agent.set_best_run(best_actions, best_score, best_death_frame)
                print(f"\n🏆 NEW BEST! Score: {best_score:.1f} (previous best: {best_score - current_score + current_score:.1f}) | Died at frame {death_frame}")
            else:
                # Show progress only if close to best or exploring
                if current_score > best_score * 0.8 or agent.new_action is not None:
                    print(f"📊 Ep {episode}: score {current_score:.1f} (best {best_score:.1f}) | died at {death_frame} | explore at {agent.explore_start}")
            
            # Manual save
            if manual_save_request:
                save_best_run(current_actions, current_score, episode, death_frame)
                agent.set_best_run(current_actions, current_score, death_frame)
                print(f"💾 Manual save: score {current_score:.1f}")
                manual_save_request = False
            
            # Log to CSV
            log_episode(episode, current_score, best_score, death_frame, 
                       agent.explore_start if agent.explore_start != -1 else -1,
                       agent.new_action if agent.new_action is not None else -1,
                       episode_duration)
            
            episode += 1
            conn.close()
            
            # Periodic checkpoint
            if episode % 50 == 0:
                save_checkpoint(episode, best_actions, best_score, best_death_frame)
                
        except socket.timeout:
            continue
        except Exception as e:
            print(f"⚠️ Error: {e}")
            try: conn.close()
            except: pass
    
    # Shutdown
    print("\n🛑 Saving final data...")
    save_checkpoint(episode, best_actions, best_score, best_death_frame)
    close_csv()
    server.close()
    print(f"\n✅ Training finished. Best score: {best_score:.1f}")
    sys.exit(0)

if __name__ == "__main__":
    main()