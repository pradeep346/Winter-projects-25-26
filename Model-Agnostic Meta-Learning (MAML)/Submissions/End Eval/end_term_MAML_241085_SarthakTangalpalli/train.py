import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the Small Neural Network
class WirelessNet(nn.Module):
    def __init__(self):
        super(WirelessNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 3 outputs for 3D coordinates (x, y, z)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Dataset Loading function
def load_tasks(filepath):
    data = np.load(filepath)
    # Convert numpy arrays to PyTorch tensors
    x_sup = torch.tensor(data['x_support'], dtype=torch.float32)
    y_sup = torch.tensor(data['y_support'], dtype=torch.float32)
    x_que = torch.tensor(data['x_query'], dtype=torch.float32)
    y_que = torch.tensor(data['y_query'], dtype=torch.float32)
    return x_sup, y_sup, x_que, y_que

# 3. Reptile Meta-Learning Implementation
def train_reptile(x_sup, y_sup, outer_iters=500, inner_steps=5, inner_lr=0.01, outer_lr=0.001):
    print("\n--- Starting Reptile Meta-Training ---")
    model = WirelessNet()
    outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    loss_fn = nn.MSELoss()
    
    num_tasks = x_sup.shape[0]
    
    for iteration in range(outer_iters):
        outer_optimizer.zero_grad()
        
        # We will accumulate the nudges (differences) from multiple tasks in a batch
        # For simplicity and speed in this small dataset, we sample a batch of 10 tasks
        batch_tasks = np.random.choice(num_tasks, size=10, replace=False)
        
        # Store original weights
        original_weights = [p.clone().detach() for p in model.parameters()]
        
        # Accumulate the sum of adapted weights
        sum_adapted_weights = [torch.zeros_like(p) for p in model.parameters()]
        
        total_inner_loss = 0.0
        
        for task_idx in batch_tasks:
            # Clone model for inner loop
            task_model = WirelessNet()
            task_model.load_state_dict(model.state_dict())
            
            # Using Adam for the inner loop as suggested in the guide
            inner_optimizer = optim.Adam(task_model.parameters(), lr=inner_lr)
            
            # Perform inner loop adaptation on Support Set
            for _ in range(inner_steps):
                preds = task_model(x_sup[task_idx])
                loss = loss_fn(preds, y_sup[task_idx])
                
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
                
            total_inner_loss += loss.item()
            
            # Accumulate adapted weights
            for i, p in enumerate(task_model.parameters()):
                sum_adapted_weights[i] += p.data
                
        # Now we apply the Reptile Nudge
        # Move master model weights towards the average of the adapted weights
        for i, p in enumerate(model.parameters()):
            avg_adapted_weight = sum_adapted_weights[i] / len(batch_tasks)
            # The "gradient" for the outer optimizer is the direction to the adapted weights
            p.grad = (original_weights[i] - avg_adapted_weight)
            
        outer_optimizer.step()
        
        if (iteration + 1) % 10 == 0:
            print(f"Reptile Iteration {iteration + 1}/{outer_iters} | Avg Inner Support Loss: {total_inner_loss/len(batch_tasks):.4f}")
            
    return model

# 4. MAML (First-Order / FOMAML) Meta-Learning Implementation
def train_maml(x_sup, y_sup, x_que, y_que, outer_iters=500, inner_steps=5, inner_lr=0.01, outer_lr=0.001):
    print("\n--- Starting MAML Meta-Training ---")
    model = WirelessNet()
    outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    loss_fn = nn.MSELoss()
    
    num_tasks = x_sup.shape[0]
    
    for iteration in range(outer_iters):
        outer_optimizer.zero_grad()
        
        batch_tasks = np.random.choice(num_tasks, size=10, replace=False)
        total_meta_loss = 0.0
        
        for task_idx in batch_tasks:
            # Clone model for inner loop
            task_model = WirelessNet()
            task_model.load_state_dict(model.state_dict())
            inner_optimizer = optim.Adam(task_model.parameters(), lr=inner_lr)
            
            # Inner Loop: Adapt on Support Set
            for _ in range(inner_steps):
                preds = task_model(x_sup[task_idx])
                loss = loss_fn(preds, y_sup[task_idx])
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
            
            # Outer Loop: Compute query loss on the ADAPTED model
            query_preds = task_model(x_que[task_idx])
            query_loss = loss_fn(query_preds, y_que[task_idx])
            total_meta_loss += query_loss.item()
            
            # FOMAML: Compute gradients of query loss w.r.t. adapted weights
            task_model.zero_grad()
            query_loss.backward()
            
            # Transfer gradients from adapted model to master model (first-order approx)
            for master_p, task_p in zip(model.parameters(), task_model.parameters()):
                if task_p.grad is not None:
                    if master_p.grad is None:
                        master_p.grad = task_p.grad.clone() / len(batch_tasks)
                    else:
                        master_p.grad += task_p.grad.clone() / len(batch_tasks)
        
        # Step master model with accumulated gradients
        outer_optimizer.step()
        
        if (iteration + 1) % 10 == 0:
            print(f"MAML Iteration {iteration + 1}/{outer_iters} | Meta (Query) Loss: {total_meta_loss/len(batch_tasks):.4f}")
            
    return model


# 5. Baseline Model — Train from Scratch (no meta-learning)
def train_baseline_on_task(x_support, y_support, x_query, y_query, train_steps=200, lr=0.01):
    """
    Trains a brand new neural network FROM SCRATCH on a single task's support set.
    This is the naive approach: no shared initialization, no meta-learning.
    Returns the MSE loss on the query set after training.
    """
    model = WirelessNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Train on the support set for 200 steps
    for step in range(train_steps):
        preds = model(x_support)
        loss = loss_fn(preds, y_support)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate on query set
    with torch.no_grad():
        query_preds = model(x_query)
        query_loss = loss_fn(query_preds, y_query).item()
    
    return query_loss

if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define absolute paths relative to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    print(f"Loading datasets from {data_dir}...")
    x_sup, y_sup, x_que, y_que = load_tasks(os.path.join(data_dir, "train_data.npz"))
    print(f"Loaded {x_sup.shape[0]} training tasks.")
    
    # Create the models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # --- Baseline Demo ---
    # Run baseline on 5 training tasks to verify it works
    print("\n--- Baseline Model Demo (train from scratch) ---")
    demo_losses = []
    for i in range(5):
        bl_loss = train_baseline_on_task(x_sup[i], y_sup[i], x_que[i], y_que[i])
        demo_losses.append(bl_loss)
        print(f"  Task {i+1}: Baseline Query Loss = {bl_loss:.4f}")
    print(f"  Avg Baseline Query Loss (5 demo tasks): {np.mean(demo_losses):.4f}")
    
    # --- Train Reptile ---
    reptile_model = train_reptile(x_sup, y_sup, outer_iters=500, inner_steps=5, inner_lr=0.02, outer_lr=0.001)
    reptile_save_path = os.path.join(models_dir, "reptile_master.pth")
    torch.save(reptile_model.state_dict(), reptile_save_path)
    print(f"Saved Reptile master weights to: {reptile_save_path}")
    
    # --- Train MAML ---
    maml_model = train_maml(x_sup, y_sup, x_que, y_que, outer_iters=500, inner_steps=5, inner_lr=0.02, outer_lr=0.001)
    maml_save_path = os.path.join(models_dir, "maml_master.pth")
    torch.save(maml_model.state_dict(), maml_save_path)
    print(f"Saved MAML master weights to: {maml_save_path}")
    
    print("\nTraining Phase Complete! Baseline verified, Reptile and MAML models saved.")
    
    print("\nTraining Phase Complete! Baseline verified, Reptile and MAML models saved.")
