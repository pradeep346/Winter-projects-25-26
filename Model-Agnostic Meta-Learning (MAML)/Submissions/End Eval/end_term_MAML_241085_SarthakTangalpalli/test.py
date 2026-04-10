import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Import model and utilities from train.py
from train import WirelessNet, load_tasks, train_baseline_on_task

def adapt_and_evaluate(model, x_support, y_support, x_query, y_query, inner_steps=5, inner_lr=0.02):
    """
    Takes a pre-trained meta-learned model, adapts it on the support set
    for a given number of inner steps, then evaluates on the query set.
    Returns (query_loss, list_of_losses_per_step).
    """
    adapted_model = WirelessNet()
    adapted_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(adapted_model.parameters(), lr=inner_lr)
    loss_fn = nn.MSELoss()
    
    losses_per_step = []
    
    # Record error BEFORE any adaptation (0-shot)
    with torch.no_grad():
        preds = adapted_model(x_query)
        initial_loss = loss_fn(preds, y_query).item()
    losses_per_step.append(initial_loss)
    
    # Adapt on support set and record query error after each step
    for step in range(inner_steps):
        preds = adapted_model(x_support)
        loss = loss_fn(preds, y_support)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate on query set after this adaptation step
        with torch.no_grad():
            query_preds = adapted_model(x_query)
            query_loss = loss_fn(query_preds, y_query).item()
        losses_per_step.append(query_loss)
    
    return losses_per_step[-1], losses_per_step

def evaluate_baseline_sweep(x_support, y_support, x_query, y_query, step_counts=[0, 1, 2, 3, 4, 5]):
    """
    Evaluate the baseline model (trained from scratch) at different adaptation step counts.
    For each step count, trains a fresh model for that many steps and evaluates.
    """
    loss_fn = nn.MSELoss()
    losses_per_step = []
    
    for steps in step_counts:
        if steps == 0:
            # Random model, no training
            model = WirelessNet()
            with torch.no_grad():
                preds = model(x_query)
                loss = loss_fn(preds, y_query).item()
            losses_per_step.append(loss)
        else:
            model = WirelessNet()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for _ in range(steps):
                preds = model(x_support)
                loss = loss_fn(preds, y_support)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                query_preds = model(x_query)
                query_loss = loss_fn(query_preds, y_query).item()
            losses_per_step.append(query_loss)
    
    return losses_per_step

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define absolute paths relative to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    print(f"Loading test datasets from {data_dir}...")
    datasets = {
        '5-shot':  load_tasks(os.path.join(data_dir, "test_data_5shot.npz")),
        '10-shot': load_tasks(os.path.join(data_dir, "test_data.npz")),
        '20-shot': load_tasks(os.path.join(data_dir, "test_data_20shot.npz")),
    }
    print(f"Loaded 5-shot, 10-shot, and 20-shot test datasets (20 tasks each).\n")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # =========================================================================
    # Load Meta-Learned Models
    # =========================================================================
    print(f"Loading pre-trained models from {models_dir}...")
    reptile_model = WirelessNet()
    reptile_model.load_state_dict(torch.load(os.path.join(models_dir, "reptile_master.pth"), weights_only=True))
    
    maml_model = WirelessNet()
    maml_model.load_state_dict(torch.load(os.path.join(models_dir, "maml_master.pth"), weights_only=True))
    
    # =========================================================================
    # Evaluate all 3 methods across 5-shot, 10-shot, 20-shot
    # =========================================================================
    adaptation_steps = 5
    step_range = list(range(adaptation_steps + 1))  # [0, 1, 2, 3, 4, 5]
    
    # Store results for the summary table
    results_table = {}
    
    # Also store per-step curves for the 10-shot dataset (used for Plot 2)
    all_reptile_steps = []
    all_maml_steps = []
    all_baseline_steps = []
    
    for shot_name, (x_sup, y_sup, x_que, y_que) in datasets.items():
        num_test_tasks = x_sup.shape[0]
        
        reptile_errors = []
        maml_errors = []
        baseline_errors = []
        
        print(f"\n{'='*60}")
        print(f"  Evaluating {shot_name} ({num_test_tasks} tasks)")
        print(f"{'='*60}")
        print(f"{'Task':<6} {'Baseline':>12} {'Reptile':>12} {'MAML':>12}")
        print("-" * 48)
        
        for t in range(num_test_tasks):
            # Reptile
            rep_final, rep_steps = adapt_and_evaluate(
                reptile_model, x_sup[t], y_sup[t], x_que[t], y_que[t],
                inner_steps=adaptation_steps, inner_lr=0.02
            )
            reptile_errors.append(rep_final)
            
            # MAML
            maml_final, maml_steps = adapt_and_evaluate(
                maml_model, x_sup[t], y_sup[t], x_que[t], y_que[t],
                inner_steps=adaptation_steps, inner_lr=0.02
            )
            maml_errors.append(maml_final)
            
            # Baseline
            bl_loss = train_baseline_on_task(x_sup[t], y_sup[t], x_que[t], y_que[t])
            baseline_errors.append(bl_loss)
            
            # Store per-step curves for 10-shot (used in Plot 2)
            if shot_name == '10-shot':
                all_reptile_steps.append(rep_steps)
                all_maml_steps.append(maml_steps)
                bl_sweep = evaluate_baseline_sweep(x_sup[t], y_sup[t], x_que[t], y_que[t], step_counts=step_range)
                all_baseline_steps.append(bl_sweep)
            
            print(f"  {t+1:<4} {bl_loss:>12.4f} {rep_final:>12.4f} {maml_final:>12.4f}")
        
        results_table[shot_name] = {
            'baseline': np.mean(baseline_errors),
            'reptile': np.mean(reptile_errors),
            'maml': np.mean(maml_errors),
        }
    
    # =========================================================================
    # Print Summary Table (matches the README format)
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"{'Method':<30} {'5-shot MSE':>12} {'10-shot MSE':>12} {'20-shot MSE':>12}")
    print("-" * 65)
    print(f"{'Baseline (from scratch)':<30} {results_table['5-shot']['baseline']:>12.4f} {results_table['10-shot']['baseline']:>12.4f} {results_table['20-shot']['baseline']:>12.4f}")
    print(f"{'MAML':<30} {results_table['5-shot']['maml']:>12.4f} {results_table['10-shot']['maml']:>12.4f} {results_table['20-shot']['maml']:>12.4f}")
    print(f"{'Reptile':<30} {results_table['5-shot']['reptile']:>12.4f} {results_table['10-shot']['reptile']:>12.4f} {results_table['20-shot']['reptile']:>12.4f}")
    print(f"{'='*65}")
    
    # =========================================================================
    # Plot 1: Training Loss Curve (reload from a quick re-run log)
    # We re-run a short meta-training to capture the loss curve
    # =========================================================================
    print("\nGenerating Plot 1: Training Loss Curves...")
    
    # Load training data for loss curve generation
    x_tr_sup, y_tr_sup, x_tr_que, y_tr_que = load_tasks(os.path.join(data_dir, "train_data.npz"))
    
    # Quick Reptile training to capture loss curve
    reptile_losses = []
    model_r = WirelessNet()
    outer_opt_r = optim.Adam(model_r.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    num_train = x_tr_sup.shape[0]
    
    for iteration in range(300):
        outer_opt_r.zero_grad()
        batch = np.random.choice(num_train, size=10, replace=False)
        orig_w = [p.clone().detach() for p in model_r.parameters()]
        sum_w = [torch.zeros_like(p) for p in model_r.parameters()]
        total_loss = 0.0
        for idx in batch:
            tm = WirelessNet()
            tm.load_state_dict(model_r.state_dict())
            iopt = optim.Adam(tm.parameters(), lr=0.02)
            for _ in range(5):
                p = tm(x_tr_sup[idx])
                l = loss_fn(p, y_tr_sup[idx])
                iopt.zero_grad()
                l.backward()
                iopt.step()
            total_loss += l.item()
            for i, param in enumerate(tm.parameters()):
                sum_w[i] += param.data
        for i, param in enumerate(model_r.parameters()):
            param.grad = (orig_w[i] - sum_w[i] / len(batch))
        outer_opt_r.step()
        reptile_losses.append(total_loss / len(batch))
    
    # Quick MAML training to capture loss curve (using corrected FOMAML)
    maml_losses = []
    model_m = WirelessNet()
    outer_opt_m = optim.Adam(model_m.parameters(), lr=0.001)
    
    for iteration in range(300):
        outer_opt_m.zero_grad()
        batch = np.random.choice(num_train, size=10, replace=False)
        total_qloss = 0.0
        for idx in batch:
            tm = WirelessNet()
            tm.load_state_dict(model_m.state_dict())
            iopt = optim.Adam(tm.parameters(), lr=0.02)
            for _ in range(5):
                p = tm(x_tr_sup[idx])
                l = loss_fn(p, y_tr_sup[idx])
                iopt.zero_grad()
                l.backward()
                iopt.step()
            qp = tm(x_tr_que[idx])
            ql = loss_fn(qp, y_tr_que[idx])
            total_qloss += ql.item()
            tm.zero_grad()
            ql.backward()
            for mp, tp in zip(model_m.parameters(), tm.parameters()):
                if tp.grad is not None:
                    if mp.grad is None:
                        mp.grad = tp.grad.clone() / len(batch)
                    else:
                        mp.grad += tp.grad.clone() / len(batch)
        outer_opt_m.step()
        maml_losses.append(total_qloss / len(batch))
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(reptile_losses, label='Reptile', color='#2196F3', linewidth=2)
    ax1.plot(maml_losses, label='MAML', color='#FF5722', linewidth=2)
    ax1.set_xlabel('Meta-Training Iteration', fontsize=13)
    ax1.set_ylabel('Loss (MSE)', fontsize=13)
    ax1.set_title('Meta-Training Loss Curves: Reptile vs MAML', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plot_loss_path = os.path.join(results_dir, 'plot_loss.png')
    fig1.savefig(plot_loss_path, dpi=150)
    print(f"Saved Plot 1 to: {plot_loss_path}")
    
    # =========================================================================
    # Plot 2: MAML vs Reptile vs Baseline Comparison (BONUS)
    # =========================================================================
    print("Generating Plot 2: Comparison Plot...")
    
    avg_baseline_curve = np.mean(all_baseline_steps, axis=0)
    avg_reptile_curve = np.mean(all_reptile_steps, axis=0)
    avg_maml_curve = np.mean(all_maml_steps, axis=0)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(step_range, avg_baseline_curve, label='Baseline (from scratch)', color='#9E9E9E', linewidth=2, marker='o')
    ax2.plot(step_range, avg_reptile_curve, label='Reptile', color='#2196F3', linewidth=2, marker='s')
    ax2.plot(step_range, avg_maml_curve, label='MAML', color='#FF5722', linewidth=2, marker='^')
    ax2.set_xlabel('Number of Adaptation Steps', fontsize=13)
    ax2.set_ylabel('Query Set MSE Loss', fontsize=13)
    ax2.set_title('Few-Shot Adaptation: MAML vs Reptile vs Baseline', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.set_xticks(step_range)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plot_comparison_path = os.path.join(results_dir, 'plot_comparison.png')
    fig2.savefig(plot_comparison_path, dpi=150)
    print(f"Saved Plot 2 to: {plot_comparison_path}")
    
    print(f"\n✓ Testing complete! Check {results_dir} folder for plots.")
