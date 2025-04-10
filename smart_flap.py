import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections

device = None
# Check if MPS (Metal Performance Shaders) is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available. Please check your PyTorch installation or use a different device.")
else:
    print("MPS is available. Using it for computations.")
    device = torch.device("mps")

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


def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0,1)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            return action


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

buffer_capacity = 10000
replay_buffer = ReplayBuffer(buffer_capacity)
