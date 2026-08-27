# main_train.py

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import DataLoader

from config import (
    SEED,
    BATCH_SIZE,
    MIN_FREQUENCY,
    VALIDATION_SIZE,
    D_MODEL,
    NUM_HEADS,
    D_FF,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    CHECKPOINT_PATH,
    EXPERIMENT_NAME,
    EXPERIMENT_DIR,
    PLOT_DIR
)

from data.data_utils import (
    build_vocab,
    create_collate_fn
)

from model.transformer import Transformer

from training.train import train_one_epoch
from training.evaluate import evaluate_model
from training.metrics import calculate_perplexity

from experiments.logger import (
    save_epoch_history,
    save_experiment_summary
)

from experiments.plotting import (
    plot_training_history
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main Training Pipeline
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)


    # --------------------------------------------------------
    # 2. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        total_gpu_memory_gb = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"GPU Memory: "
            f"{total_gpu_memory_gb:.2f} GB"
        )


    # --------------------------------------------------------
    # 3. Experiment information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT")
    print("=" * 70)

    print(
        "Experiment name:",
        EXPERIMENT_NAME
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Experiment directory:",
        EXPERIMENT_DIR
    )

    print(
        "Plot directory:",
        PLOT_DIR
    )


    # --------------------------------------------------------
    # 4. Create output directories
    # --------------------------------------------------------

    checkpoint_directory = os.path.dirname(
        CHECKPOINT_PATH
    )

    if checkpoint_directory:

        os.makedirs(
            checkpoint_directory,
            exist_ok=True
        )

    os.makedirs(
        EXPERIMENT_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 5. Output file paths
    # --------------------------------------------------------

    history_path = os.path.join(
        EXPERIMENT_DIR,
        f"{EXPERIMENT_NAME}_history.csv"
    )

    summary_path = os.path.join(
        EXPERIMENT_DIR,
        "results.csv"
    )


    # --------------------------------------------------------
    # 6. Load dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    dataset = load_dataset(
        "Helsinki-NLP/opus_books",
        "de-en"
    )

    full_dataset = dataset["train"]

    print(
        "Total examples:",
        len(full_dataset)
    )


    # --------------------------------------------------------
    # 7. Train / validation split
    # --------------------------------------------------------

    split_data = (
        full_dataset
        .train_test_split(
            test_size=VALIDATION_SIZE,
            seed=SEED
        )
    )

    train_dataset = split_data["train"]
    val_dataset = split_data["test"]

    print(
        "Training examples:",
        len(train_dataset)
    )

    print(
        "Validation examples:",
        len(val_dataset)
    )


    # --------------------------------------------------------
    # 8. Build vocabulary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)

    german_vocab, english_vocab = build_vocab(
        train_dataset,
        min_frequency=MIN_FREQUENCY
    )

    print(
        "German vocabulary:",
        len(german_vocab)
    )

    print(
        "English vocabulary:",
        len(english_vocab)
    )


    # --------------------------------------------------------
    # 9. Collate function
    # --------------------------------------------------------

    collate_fn = create_collate_fn(
        german_vocab,
        english_vocab
    )


    # --------------------------------------------------------
    # 10. DataLoaders
    # --------------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_fn,

        num_workers=2,

        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_fn,

        num_workers=2,

        pin_memory=pin_memory
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )


    # --------------------------------------------------------
    # 11. Create model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING TRANSFORMER")
    print("=" * 70)

    model = Transformer(

        src_vocab_size=
            len(german_vocab),

        tgt_vocab_size=
            len(english_vocab),

        d_model=
            D_MODEL,

        num_heads=
            NUM_HEADS,

        d_ff=
            D_FF,

        num_encoder_layers=
            NUM_ENCODER_LAYERS,

        num_decoder_layers=
            NUM_DECODER_LAYERS,

        dropout=
            DROPOUT
    )

    model = model.to(device)


    # --------------------------------------------------------
    # 12. Parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )


    # --------------------------------------------------------
    # 13. Loss
    # --------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=
            english_vocab["<pad>"]
    )


    # --------------------------------------------------------
    # 14. Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # 15. Show configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print("D_MODEL:", D_MODEL)
    print("HEADS:", NUM_HEADS)
    print("D_FF:", D_FF)

    print(
        "ENCODER LAYERS:",
        NUM_ENCODER_LAYERS
    )

    print(
        "DECODER LAYERS:",
        NUM_DECODER_LAYERS
    )

    print(
        "DROPOUT:",
        DROPOUT
    )

    print(
        "BATCH SIZE:",
        BATCH_SIZE
    )

    print(
        "LEARNING RATE:",
        LEARNING_RATE
    )

    print(
        "EPOCHS:",
        EPOCHS
    )


    # --------------------------------------------------------
    # 16. Experiment tracking variables
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = None

    best_perplexity = None

    training_start_time = (
        time.time()
    )


    # --------------------------------------------------------
    # 17. Reset peak GPU stats
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats()


    # --------------------------------------------------------
    # 18. Training loop
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        epoch_start_time = (
            time.time()
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(

            model=model,

            train_loader=train_loader,

            optimizer=optimizer,

            loss_fn=loss_fn,

            device=device
        )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        val_loss = evaluate_model(

            model=model,

            val_loader=val_loader,

            loss_fn=loss_fn,

            device=device
        )


        # ----------------------------------------------------
        # Perplexity
        # ----------------------------------------------------

        perplexity = calculate_perplexity(
            val_loss
        )


        # ----------------------------------------------------
        # Epoch duration
        # ----------------------------------------------------

        epoch_time_min = (
            time.time()
            - epoch_start_time
        ) / 60


        # ----------------------------------------------------
        # Save epoch history
        # ----------------------------------------------------

        save_epoch_history(

            file_path=
                history_path,

            epoch=
                epoch + 1,

            train_loss=
                train_loss,

            val_loss=
                val_loss,

            perplexity=
                perplexity,

            epoch_time_min=
                epoch_time_min
        )


        # ----------------------------------------------------
        # Print epoch
        # ----------------------------------------------------

        print(

            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{EPOCHS:02d} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Val Loss: "
            f"{val_loss:.4f} | "

            f"PPL: "
            f"{perplexity:.2f} | "

            f"Time: "
            f"{epoch_time_min:.2f} min"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            best_epoch = (
                epoch + 1
            )

            best_perplexity = (
                perplexity
            )


            checkpoint = {

                "epoch":
                    epoch + 1,

                "experiment_name":
                    EXPERIMENT_NAME,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "german_vocab":
                    german_vocab,

                "english_vocab":
                    english_vocab,

                "config": {

                    "d_model":
                        D_MODEL,

                    "num_heads":
                        NUM_HEADS,

                    "d_ff":
                        D_FF,

                    "num_encoder_layers":
                        NUM_ENCODER_LAYERS,

                    "num_decoder_layers":
                        NUM_DECODER_LAYERS,

                    "dropout":
                        DROPOUT,

                    "learning_rate":
                        LEARNING_RATE,

                    "batch_size":
                        BATCH_SIZE
                },

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "perplexity":
                    perplexity,

                "total_parameters":
                    total_parameters
            }


            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )


            print(
                "Best model saved:",
                CHECKPOINT_PATH
            )


    # --------------------------------------------------------
    # 19. Total training duration
    # --------------------------------------------------------

    total_time_min = (
        time.time()
        - training_start_time
    ) / 60


    # --------------------------------------------------------
    # 20. GPU peak memory
    # --------------------------------------------------------

    peak_gpu_memory_gb = None

    if device.type == "cuda":

        peak_gpu_memory_gb = (

            torch.cuda
            .max_memory_allocated()

            / 1024**3
        )


    # --------------------------------------------------------
    # 21. Config for summary file
    # --------------------------------------------------------

    experiment_config = {

        "d_model":
            D_MODEL,

        "num_heads":
            NUM_HEADS,

        "d_ff":
            D_FF,

        "num_encoder_layers":
            NUM_ENCODER_LAYERS,

        "num_decoder_layers":
            NUM_DECODER_LAYERS,

        "dropout":
            DROPOUT,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE
    }


    # --------------------------------------------------------
    # 22. Save experiment summary
    # --------------------------------------------------------

    save_experiment_summary(

        file_path=
            summary_path,

        experiment_name=
            EXPERIMENT_NAME,

        config=
            experiment_config,

        best_epoch=
            best_epoch,

        best_val_loss=
            best_val_loss,

        best_perplexity=
            best_perplexity,

        total_time_min=
            total_time_min,

        checkpoint_path=
            CHECKPOINT_PATH,

        greedy_bleu=None,

        beam_bleu=None
    )


    # --------------------------------------------------------
    # 23. Generate plots
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_paths = plot_training_history(

        history_csv_path=
            history_path,

        output_dir=
            PLOT_DIR,

        experiment_name=
            EXPERIMENT_NAME
    )


    # --------------------------------------------------------
    # 24. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation loss:",
        round(
            best_val_loss,
            4
        )
    )

    print(
        "Best perplexity:",
        round(
            best_perplexity,
            2
        )
    )

    print(
        "Total training time:",
        round(
            total_time_min,
            2
        ),
        "minutes"
    )

    if peak_gpu_memory_gb is not None:

        print(
            "Peak GPU memory:",
            round(
                peak_gpu_memory_gb,
                2
            ),
            "GB"
        )


    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    print(
        "\nEpoch history:"
    )

    print(
        history_path
    )


    print(
        "\nExperiment summary:"
    )

    print(
        summary_path
    )


    print(
        "\nLoss plot:"
    )

    print(
        plot_paths["loss_plot"]
    )


    print(
        "\nPerplexity plot:"
    )

    print(
        plot_paths[
            "perplexity_plot"
        ]
    )


    print(
        "\nEpoch-time plot:"
    )

    print(
        plot_paths["time_plot"]
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()