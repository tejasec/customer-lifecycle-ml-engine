from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ["No Churn", "Churn"],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_loss_curves(
    loss_dict: dict,
    title: str = "Optimizer Loss Trajectories",
    save_path: Optional[Path] = None
) -> None:
    plt.figure(figsize=(9, 5))
    for opt_name, losses in loss_dict.items():
        plt.plot(losses, label=opt_name, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_cluster_scatter(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Cluster Visualization",
    save_path: Optional[Path] = None
) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_coords, y_coords, c=cluster_labels, cmap="viridis", alpha=0.7, edgecolors="k")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
