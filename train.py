import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from environment import GridWorld
from a3c_agent import A3C

env = GridWorld()
model = A3C()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

episodes = 500
rewards = []

for episode in range(episodes):

    # Decaying exploration
    epsilon = max(0.05, 0.2 - episode / 500)

    state = env.reset()
    state = torch.FloatTensor(state) / 6.0

    total_reward = 0

    for step in range(50):
        policy, value = model(state)
        probs = torch.softmax(policy, dim=0)

        # Exploration vs Exploitation
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = torch.multinomial(probs, 1).item()

        next_state, reward, done = env.step(action)
        next_state = torch.FloatTensor(next_state) / 6.0

        total_reward += reward

        _, next_value = model(next_state)

        target = reward + 0.99 * next_value * (1 - int(done))
        advantage = target - value

        actor_loss = -torch.log(probs[action] + 1e-8) * advantage
        critic_loss = advantage.pow(2)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if done:
            break

    rewards.append(total_reward)

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Save model
torch.save(model.state_dict(), "a3c_model.pth")

# Plot graph
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance")
plt.show()