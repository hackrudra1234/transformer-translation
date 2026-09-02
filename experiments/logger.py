import csv
import os


def save_epoch_history(
    file_path,
    epoch,
    train_loss,
    val_loss,
    perplexity,
    epoch_time_min
):
    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    file_exists = os.path.exists(file_path)

    with open(
        file_path,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "perplexity",
                    "epoch_time_min"
                ]
            )

        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss,
                perplexity,
                epoch_time_min
            ]
        )
def save_experiment_summary(
    file_path,
    experiment_name,
    config,
    best_epoch,
    best_val_loss,
    best_perplexity,
    total_time_min,
    checkpoint_path,
    greedy_bleu=None,
    beam_bleu=None,
):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    file_exists = os.path.exists(file_path)

    with open(
        file_path,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow(
                [
                    "experiment",
                    "d_model",
                    "num_heads",
                    "d_ff",
                    "encoder_layers",
                    "decoder_layers",
                    "dropout",
                    "learning_rate",
                    "batch_size",
                    "initialization",
                     "weight_tying",
                    "lr_schedule",
                    "warmup_steps",
                    "label_smoothing",
                    "best_epoch",
                    "best_val_loss",
                    "best_perplexity",
                    "greedy_bleu",
                    "beam_bleu",
                    "total_time_min",
                    "checkpoint_path"
                    
                ]
            )

        writer.writerow(
            [
                experiment_name,
                config["d_model"],
                config["num_heads"],
                config["d_ff"],
                config["num_encoder_layers"],
                config["num_decoder_layers"],
                config["dropout"],
                config["learning_rate"],
                config["batch_size"],
                config["initialization"],
                config["weight_tying"],
                config["lr_schedule"],
                config["warmup_steps"],
                config["label_smoothing"],
                best_epoch,
                best_val_loss,
                best_perplexity,
                greedy_bleu,
                beam_bleu,
                total_time_min,
                checkpoint_path
            ]
        )