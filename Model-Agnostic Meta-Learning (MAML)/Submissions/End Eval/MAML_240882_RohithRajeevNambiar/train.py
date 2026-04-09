import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tasks.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

data = np.load(DATA_PATH)

train_Xs = data["train_Xs"]
train_Ys = data["train_Ys"]
train_Xq = data["train_Xq"]
train_Yq = data["train_Yq"]

m = train_Xs.shape[-1]

model = nn.Sequential(
    nn.Linear(m, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

inner_lr = 0.01
inner_steps = 3
meta_iterations = 200

loss_history = []

for iteration in range(meta_iterations):
    meta_loss = 0

    for i in range(len(train_Xs)):
        Xs = torch.tensor(train_Xs[i], dtype=torch.float32)
        Ys = torch.tensor(train_Ys[i], dtype=torch.float32)
        Xq = torch.tensor(train_Xq[i], dtype=torch.float32)
        Yq = torch.tensor(train_Yq[i], dtype=torch.float32)

        adapted_model = copy.deepcopy(model)

        for _ in range(inner_steps):
            pred = adapted_model(Xs)
            loss = loss_fn(pred, Ys)

            grads = torch.autograd.grad(loss, adapted_model.parameters())

            for p, g in zip(adapted_model.parameters(), grads):
                p.data -= inner_lr * g

        loss_q = loss_fn(adapted_model(Xq), Yq)
        meta_loss += loss_q

    meta_loss /= len(train_Xs)
    loss_history.append(meta_loss.item())

    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {meta_loss.item():.4f}")

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
torch.save(model.state_dict(), MODEL_PATH)

# Plot
plt.figure()
plt.plot(loss_history)
plt.xlabel("Meta-Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "plot_loss.png"))
plt.close()

print("Training complete.")