import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_history(
    history_csv_path,
    output_dir,
    experiment_name
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    history = pd.read_csv(
        history_csv_path
    )

    # --------------------------------------------------------
    # Loss plot
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Training vs Validation Loss\n{experiment_name}"
    )

    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(
        output_dir,
        f"{experiment_name}_loss.png"
    )

    plt.savefig(
        loss_plot_path,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # Perplexity plot
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        history["epoch"],
        history["perplexity"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title(
        f"Validation Perplexity\n{experiment_name}"
    )

    plt.grid(True)

    perplexity_plot_path = os.path.join(
        output_dir,
        f"{experiment_name}_perplexity.png"
    )

    plt.savefig(
        perplexity_plot_path,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # Epoch time plot
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        history["epoch"],
        history["epoch_time_min"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Time (minutes)")
    plt.title(
        f"Epoch Training Time\n{experiment_name}"
    )

    plt.grid(True)

    time_plot_path = os.path.join(
        output_dir,
        f"{experiment_name}_time.png"
    )

    plt.savefig(
        time_plot_path,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "loss_plot": loss_plot_path,
        "perplexity_plot": perplexity_plot_path,
        "time_plot": time_plot_path
    }