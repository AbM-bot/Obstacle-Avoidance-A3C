import torch
from environment import GridWorld
from a3c_agent import A3C

env = GridWorld()
model = A3C()
model.load_state_dict(torch.load("a3c_model.pth"))

state = env.reset()
state = torch.FloatTensor(state)

for step in range(20):
    policy, _ = model(state)

    action = torch.argmax(policy).item()

    next_state, reward, done = env.step(action)
    state = torch.FloatTensor(next_state)

    print("Step:", step, "State:", next_state, "Reward:", reward)

    if done:
        print("Finished")
        break