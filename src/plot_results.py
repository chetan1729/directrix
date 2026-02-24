import matplotlib.pyplot as plt
import re
import os

# Ensure the logs exist
train_log_path = "logs/training.log"
val_log_path = "logs/validation.log"

def parse_logs():
    steps, losses = [], []
    val_steps, val_accs = [], []

    # Parse Training Loss
    if os.path.exists(train_log_path):
        with open(train_log_path, "r") as f:
            for line in f:
                # Matches patterns like "Step 100: loss=0.5432" or similar
                step_match = re.search(r"Step (\d+)", line)
                loss_match = re.search(r"Loss:? (\d+\.\d+)", line, re.IGNORECASE)
                if step_match and loss_match:
                    steps.append(int(step_match.group(1)))
                    losses.append(float(loss_match.group(1)))

    # Parse Validation Accuracy
    if os.path.exists(val_log_path):
        with open(val_log_path, "r") as f:
            for line in f:
                step_match = re.search(r"Step (\d+)", line)
                acc_match = re.search(r"Accuracy:? (\d+\.\d+)", line, re.IGNORECASE)
                if step_match and acc_match:
                    val_steps.append(int(step_match.group(1)))
                    val_accs.append(float(acc_match.group(1)))
    
    return steps, losses, val_steps, val_accs

def generate_plots():
    steps, losses, val_steps, val_accs = parse_logs()

    if not steps:
        print("No training data found in logs. Check formatting.")
        return

    # Create Figure
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Training Loss
    ax1.plot(steps, losses, color='#1f77b4', label='Training Loss', linewidth=2)
    ax1.set_title("DPO Training Convergence", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()

    # Plot 2: Validation Accuracy
    if val_steps:
        ax2.plot(val_steps, val_accs, color='#ff7f0e', marker='o', label='Pairwise Accuracy')
        ax2.axhline(y=0.622, color='r', linestyle='--', label='Final Accuracy (62.2%)')
        ax2.set_title("Alignment Preference Accuracy", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Steps", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_ylim(0.45, 0.70)
        ax2.legend()

    plt.tight_layout()
    plt.savefig("logs/project_metrics.png")
    print("Plots saved to logs/project_metrics.png")

if __name__ == "__main__":
    generate_plots()