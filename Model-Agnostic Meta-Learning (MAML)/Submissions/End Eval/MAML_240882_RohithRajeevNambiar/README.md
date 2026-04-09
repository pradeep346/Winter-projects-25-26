# Meta-Learning for Wireless Systems (Channel Estimation)

## 1. What did I build?

This project implements a meta-learning model using **First-Order MAML (FOMAML)** for the task of **wireless channel estimation**.
The model learns a shared initialization that can quickly adapt to a new wireless environment using only a few samples.

---

## 2. How to set it up

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term
pip install -r requirements.txt
```

---

## 3. How to generate data

```bash
python generate_data.py
```

This script creates synthetic wireless tasks.
Each task simulates a different wireless environment with:

* Random channel (H)
* Random SNR (noise level)

Each task contains:

* Support set (5 samples) for adaptation
* Query set (50 samples) for evaluation

It generates:

* 100 training tasks
* 20 test tasks

All data is saved in `data/tasks.npz`.

---

## 4. How to train

```bash
python train.py
```

This script:

* Trains a meta-learning model using FOMAML
* Uses:

  * Inner learning rate = 0.01
  * 3 inner steps
  * 200 meta-iterations

It also:

* Trains a baseline model (from scratch per task)
* Saves the trained model to `model.pth`
* Saves training loss plot to `results/plot_loss.png`

---

## 5. How to test

```bash
python test.py
```

This script:

* Evaluates MAML on unseen tasks
* Adapts using 1, 3, 5, and 10 steps
* Compares with baseline performance

It outputs:

* Average loss values
* Comparison plot saved as `results/plot_comparison.png`

---

## 6. Results

| Method                  | Performance |
| ----------------------- | ----------- |
| Baseline (from scratch) | Higher loss |
| MAML (FOMAML)           | Lower loss  |

MAML consistently achieves lower error with fewer adaptation steps, demonstrating faster learning and better generalization across wireless environments.

---

## Project Structure

```
end_term/
├── README.md
├── requirements.txt
├── generate_data.py
├── train.py
├── test.py
├── data/
│   └── tasks.npz
├── results/
│   ├── plot_loss.png
│   └── plot_comparison.png
├── model.pth
```

---

## Notes

* The model uses a simple 3-layer neural network (64 hidden units).
* Tasks simulate wireless environments using a linear channel model.
* MAML enables fast adaptation using very few samples compared to training from scratch.
