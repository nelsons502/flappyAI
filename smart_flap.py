import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import socket
import sys
import math

device = None
# Check if MPS (Metal Performance Shaders) is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available. Please check your PyTorch installation or use a different device.")
else:
    print("MPS is available. Using it for computations.")
    device = torch.device("mps")

mode = None

BATCH_SIZE = 32
BUFFER_CAPACITY = 10000
PORT = 12345
HOST = "localhost"
# epsilon decay parameters
EPSILON_START = 1.0
EPSILON_END = 0.001 # potentially increase to 0.01
EPSILON_DECAY = 100000 # potntially increase
epsilon = EPSILON_START
# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
NUMBER_OF_EPISODES = 100000 # potentially increase
TARGET_UPDATE_FREQ = 1000
step_count = 0

# set up the AI model
class Flap_DPN(nn.Module):
    def __init__(self, input_size, output_size) :
        super(Flap_DPN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # First layer
        self.fc2 = nn.Linear(128, 64)         # Second layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer
        self.relu = nn.ReLU()                  # Activation function

    def forward(self, x):
        x = self.fc1(x)                        # First layer
        x = self.relu(x)                       # Activation
        x = self.fc2(x)                        # Second layer
        x = self.relu(x)                       # Activation
        x = self.fc3(x)                        # Output layer
        return x

# set up replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )
    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net=None): # need to pass policy_net for training mode
    global mode, epsilon
    if mode == "random" or policy_net is None:
        return random.choice([0]*29 + [1])
    if random.random() < epsilon:
        return random.choice([0]*29 + [1])
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            return action
        
def insert_data(state, action, reward, next_state, done):
    global step_count
    # Add the transition to the replay buffer
    replay_buffer.add(state, action, reward, next_state, done)

    # Sample a batch of transitions from the replay buffer
    if len(replay_buffer) > BATCH_SIZE:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        # Compute the target Q-values
        target_q_values = target_net(next_states).max(1)[0].detach()
        not_dones = 1.0 - dones
        expected_q_values = rewards + not_dones * GAMMA * target_q_values

        # Compute the current Q-values
        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute the loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += 1
        # Update epsilon
        update_epsilon()
        # Update the target network
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Target network updated.")
            print("Step count:", step_count)
            print("Loss:", loss.item())
            # Save models
            torch.save(policy_net.state_dict(), "flap_dqn.pth")
            torch.save(target_net.state_dict(), "flap_dqn_target.pth")
            print("Models saved.")
    
def receive_experience(conn, state_size=14): # state_size should be 14
    data = conn.recv(1024).decode('utf-8')
    floats = list(map(float, data.strip().split(',')))
    state = floats[:state_size]
    action = int(floats[state_size])
    reward = floats[state_size + 1]
    next_state = floats[state_size + 2:2 * state_size + 2]
    done = bool(floats[-1])
    return state, action, reward, next_state, done

def receive_state(conn):
    state = conn.recv(1024).decode('utf-8')
    state = list(map(float, state.strip().split(',')))
    return state

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("AI server listening on port", PORT)

    conn, addr = server_socket.accept()
    print("Connection from", addr)
    episode_reward = 0 # track total reward for the episode
    episode = 0 # track the episode number
    printed = False # track if the episode reward has been printed
    try:
        while True:
            state = receive_state(conn)
            if not state:
                break
            action = select_action(state, policy_net)
            conn.sendall(str(action).encode('utf-8'))

            # recieve state and reward
            data = receive_experience(conn)
            if not data:
                break
            state, action, reward, next_state, done = data
            # record state, action, reward, next_state, done
            insert_data(state, action, reward, next_state, done)

            episode_reward += reward
            if done and not printed:
                printed = True
                print(f"Episode {episode}: Total Reward = {episode_reward}")
                episode_reward = 0
                episode += 1
                if episode > NUMBER_OF_EPISODES:
                    print("Training complete.")
                    break
            if not done and printed:
                printed = False

    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
        server_socket.close()

def update_epsilon():
    global epsilon, step_count
    # Update epsilon using exponential decay
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-step_count / EPSILON_DECAY)
    # Print the current epsilon value
    if step_count % 500 == 0:
        print(f"Epsilon: {epsilon:.4f}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    policy_net = Flap_DPN(14, 2).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # Initialize target_net and sync it to policy_net
    target_net = Flap_DPN(14, 2).to(device)

    # Optionally load saved weights
    if mode == "continue":
        try:
            policy_net.load_state_dict(torch.load("flap_dqn.pth"))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Starting training from scratch.")
        try:
            target_net.load_state_dict(torch.load("flap_dqn_target.pth"))
            print("Target model loaded successfully.")
        except FileNotFoundError:
            print("No saved target model found. Initializing from policy net.")
            target_net.load_state_dict(policy_net.state_dict())
    else:
        target_net.load_state_dict(policy_net.state_dict())
    # Always set to train mode
    policy_net.train()

    # start the server (and thus the game)
    start_server()