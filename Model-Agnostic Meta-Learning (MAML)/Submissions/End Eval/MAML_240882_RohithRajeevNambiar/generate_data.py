import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
NUM_TRAIN_TASKS = 100
NUM_TEST_TASKS = 20

L = 3
SUPPORT = 5
QUERY = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_PATH = os.path.join(DATA_DIR, "tasks.npz")

os.makedirs(DATA_DIR, exist_ok=True)

def snr_to_noise(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    return 1 / np.sqrt(snr_linear)

def generate_task():
    H = np.random.randn(L, 1)
    snr_db = np.random.uniform(0, 20)
    noise_std = snr_to_noise(snr_db)

    Xs = np.random.randn(SUPPORT, L)
    Ys = Xs @ H + noise_std * np.random.randn(SUPPORT, 1)

    Xq = np.random.randn(QUERY, L)
    Yq = Xq @ H + noise_std * np.random.randn(QUERY, 1)

    return Xs, Ys, Xq, Yq

def generate_dataset(num_tasks):
    Xs_list, Ys_list, Xq_list, Yq_list = [], [], [], []

    for _ in range(num_tasks):
        Xs, Ys, Xq, Yq = generate_task()
        Xs_list.append(Xs)
        Ys_list.append(Ys)
        Xq_list.append(Xq)
        Yq_list.append(Yq)

    return (
        np.array(Xs_list),
        np.array(Ys_list),
        np.array(Xq_list),
        np.array(Yq_list),
    )

def main():
    print("Generating training tasks...")
    train_Xs, train_Ys, train_Xq, train_Yq = generate_dataset(NUM_TRAIN_TASKS)

    print("Generating test tasks...")
    test_Xs, test_Ys, test_Xq, test_Yq = generate_dataset(NUM_TEST_TASKS)

    np.savez(
        SAVE_PATH,
        train_Xs=train_Xs,
        train_Ys=train_Ys,
        train_Xq=train_Xq,
        train_Yq=train_Yq,
        test_Xs=test_Xs,
        test_Ys=test_Ys,
        test_Xq=test_Xq,
        test_Yq=test_Yq,
    )

    print(f"Saved dataset to {SAVE_PATH}")

if __name__ == "__main__":
    main()