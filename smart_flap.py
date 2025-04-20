import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import socket
import sys

device = None
# Check if MPS (Metal Performance Shaders) is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available. Please check your PyTorch installation or use a different device.")
else:
    print("MPS is available. Using it for computations.")
    device = torch.device("mps")

mode = None


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


def select_action(state, policy_net=None, epsilon=0.1): # need to pass policy_net for training mode
    if mode == "random" or policy_net is None:
        return random.choice([0]*20 + [1])
    if random.random() < epsilon:
        return random.randint(0,1)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            return action
        
def insert_data(state, action, reward, next_state, done):
    # Add the transition to the replay buffer
    replay_buffer.add(state, action, reward, next_state, done)

    # Sample a batch of transitions from the replay buffer
    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Compute the target Q-values
        target_q_values = policy_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * gamma * target_q_values

        # Compute the current Q-values
        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute the loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def start_server():
    host = "localhost"
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("AI server listening on port", port)

    conn, addr = server_socket.accept()
    print("Connection from", addr)
    try:
        while True:
            data = conn.recv(2048).decode('utf-8')
            if not data:
                break

            state = list(map(float, data.split(',')))
            print("Received state:", state)

            action = select_action(state)
            print("Selected action:", action)
            
            conn.sendall(str(action).encode('utf-8'))

            # recieve state and reward
            data = conn.recv(2048).decode('utf-8')
            if not data:
                break
            print("Received data:", data)

            state, action, reward, next_state, done = data.split(',')
            # record state, action, reward, next_state, done




    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
        server_socket.close()



buffer_capacity = 10000
replay_buffer = ReplayBuffer(buffer_capacity)
policy_net = Flap_DPN(8, 2).to(device)

if __name__ == "__main__":
    if len(sys.argv) > 1: #options should be "random", "train", "eval"
        mode = sys.argv[1]
    start_server()