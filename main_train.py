# main_train.py

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchgen import model
from data.bpe_tokenizers import load_bpe_tokenizer,train_bpe_tokenizer
from data.data_utils import create_bpe_collate_fn
from inference.greedy import greedy_decode_bpe
from inference.beam import beam_search_decode_bpe

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
    MAX_LEN,
    BEAM_SIZE,
    LENGTH_PENALTY_ALPHA,
    BLEU_MAX_EXAMPLES,
    CHECKPOINT_PATH,
    EXPERIMENT_NAME,
    EXPERIMENT_DIR,
    PLOT_DIR,
    INITIALIZATION,
    LR_SCHEDULE,
    WARMUP_STEPS,
    LABEL_SMOOTHING,
    WEIGHT_TYING,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    BPE_VOCAB_SIZE
)

from data.data_utils import (
    build_vocab,
    create_collate_fn
)

from model.transformer import Transformer

from training.train import train_one_epoch
from training.evaluate import evaluate_model

from training.metrics import (
    calculate_perplexity,
    evaluate_bleu_bpe
)

from inference.translate import (
    translate_sentence
)

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
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Seed
    # --------------------------------------------------------

    set_seed(SEED)

    def initialize_weights(model):

        if INITIALIZATION == "xavier":

            for parameter in model.parameters():

                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

        elif INITIALIZATION == "default":

        # Keep PyTorch default initialization
            pass

        else:

            raise ValueError(
            f"Unknown initialization: {INITIALIZATION}"
            )

    def transformer_lr_lambda(step):

        step = max(step, 1)

        return (
        D_MODEL ** (-0.5)
        *
        min(
            step ** (-0.5),
            step
            * WARMUP_STEPS ** (-1.5)
        )
    )
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
        "Checkpoint path:",
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
    # 5. Output paths
    # --------------------------------------------------------

    history_path = os.path.join(
        EXPERIMENT_DIR,
        f"{EXPERIMENT_NAME}_history.csv"
    )

    summary_path = os.path.join(
        EXPERIMENT_DIR,
        "results_v6.csv"
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

    split_data = full_dataset.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SEED
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

    # german_vocab, english_vocab = build_vocab(
    #     train_dataset,
    #     min_frequency=MIN_FREQUENCY
    # )

    # print(
    #     "German vocab:",
    #     len(german_vocab)
    # )

    # print(
    #     "English vocab:",
    #     len(english_vocab)
    # )
    import os

    BPE_MODEL_PREFIX = (
    f"data/bpe_shared_{BPE_VOCAB_SIZE // 1000}k"
)

    BPE_MODEL_PATH = (
    BPE_MODEL_PREFIX + ".model"
)

    if not os.path.exists(BPE_MODEL_PATH):

        print(
        f"BPE tokenizer not found. "
        f"Training {BPE_VOCAB_SIZE} vocabulary tokenizer..."
    )

    train_bpe_tokenizer(
        train_dataset=train_dataset,
        model_prefix=BPE_MODEL_PREFIX,
        vocab_size=BPE_VOCAB_SIZE
    )

    tokenizer = load_bpe_tokenizer(
    BPE_MODEL_PATH
)

    print(
    "Actual BPE vocabulary size:",
    tokenizer.get_piece_size()
)
    tokenizer = load_bpe_tokenizer(
    BPE_MODEL_PATH
)

    vocab_size = tokenizer.get_piece_size()

    print("BPE vocabulary size:", vocab_size)


    # --------------------------------------------------------
    # 9. Collate function
    # --------------------------------------------------------

    # collate_fn = create_collate_fn(
    #     german_vocab,
    #     english_vocab
    # )

    collate_fn = create_bpe_collate_fn(
    tokenizer
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
    # 11. Model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING TRANSFORMER")
    print("=" * 70)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        weight_tying=WEIGHT_TYING
    )
    initialize_weights(model)
    model = model.to(device)


    # --------------------------------------------------------
    # 12. Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
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
        ignore_index=tokenizer.pad_id(),
        label_smoothing=LABEL_SMOOTHING
    )


    # --------------------------------------------------------
    # 14. Optimizer
    # --------------------------------------------------------

    if LR_SCHEDULE == "transformer":

        optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

        scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=transformer_lr_lambda
        )
    )

    elif LR_SCHEDULE == "constant":

        optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS
    )

        scheduler = None

    else:
        raise ValueError(
        f"Unknown LR schedule: {LR_SCHEDULE}"
    )


    # --------------------------------------------------------
    # 15. Display configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print("D_MODEL:", D_MODEL)
    print("NUM_HEADS:", NUM_HEADS)
    print("D_FF:", D_FF)

    print(
        "ENCODER LAYERS:",
        NUM_ENCODER_LAYERS
    )

    print(
        "DECODER LAYERS:",
        NUM_DECODER_LAYERS
    )

    print("DROPOUT:", DROPOUT)
    print("BATCH SIZE:", BATCH_SIZE)
    print( "INITIALIZATION:",INITIALIZATION)
    print("LR SCHEDULE:",LR_SCHEDULE)
    print("WARMUP STEPS:",WARMUP_STEPS)
    print("LEARNING RATE:", LEARNING_RATE)
    print("EPOCHS:", EPOCHS)
    print("LABEL SMOOTHING:", LABEL_SMOOTHING)
    print("WEIGHT TYING:",WEIGHT_TYING)
    print("ADAM BETA1:", ADAM_BETA1)
    print("ADAM BETA2:", ADAM_BETA2)
    print("ADAM EPS:", ADAM_EPS)

    print(
        "BLEU MAX EXAMPLES:",
        BLEU_MAX_EXAMPLES
    )

    print(
        "BEAM SIZE:",
        BEAM_SIZE
    )

    print(
        "LENGTH PENALTY:",
        LENGTH_PENALTY_ALPHA
    )


    # --------------------------------------------------------
    # 16. Tracking variables
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = None
    best_perplexity = None

    training_start_time = time.time()


    # --------------------------------------------------------
    # 17. Reset GPU memory tracker
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


    # --------------------------------------------------------
    # 18. Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        epoch_start_time = time.time()


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler
        )


        # ----------------------------------------------------
        # Validation
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
        # Epoch time
        # ----------------------------------------------------

        epoch_time_min = (
            time.time()
            - epoch_start_time
        ) / 60


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        save_epoch_history(
            file_path=history_path,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            epoch_time_min=epoch_time_min
        )


        # ----------------------------------------------------
        # Display epoch
        # ----------------------------------------------------
        current_lr = optimizer.param_groups[0]["lr"]
        print(
              f"Epoch {epoch + 1:02d}/{EPOCHS:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"PPL: {perplexity:.2f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time_min:.2f} min"
                            )


        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch + 1

            best_perplexity = perplexity


            checkpoint = {

                "epoch":
                    epoch + 1,

                "experiment_name":
                    EXPERIMENT_NAME,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "bpe_model_path":
                    BPE_MODEL_PATH,

                "bpe_vocab_size":
                tokenizer.get_piece_size(),


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
                        BATCH_SIZE,
                "weight_tying": WEIGHT_TYING,
                "adam_beta1": ADAM_BETA1,
                "adam_beta2": ADAM_BETA2,
                "adam_eps": ADAM_EPS,
                 "tokenizer_type":
                "sentencepiece_bpe",

                "bpe_vocab_size":
                tokenizer.get_piece_size(),
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
    # 19. Training time
    # --------------------------------------------------------

    total_time_min = (
        time.time()
        - training_start_time
    ) / 60


    # --------------------------------------------------------
    # 20. Peak GPU memory
    # --------------------------------------------------------

    peak_gpu_memory_gb = None

    if device.type == "cuda":

        peak_gpu_memory_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )


    # --------------------------------------------------------
    # 21. Reload best checkpoint
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    best_checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Loaded best model from epoch:",
        best_checkpoint["epoch"]
    )


    # --------------------------------------------------------
    # 22. Greedy BLEU
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GREEDY BLEU")
    print("=" * 70)

    greedy_bleu_start = time.time()

    greedy_bleu = evaluate_bleu_bpe(
        model=model,
        dataset_subset=val_dataset,
        tokenizer=tokenizer,
        translate_fn=greedy_decode_bpe,
        method="greedy",
        max_examples=BLEU_MAX_EXAMPLES,
        max_len=MAX_LEN,
        beam_size=BEAM_SIZE,
        alpha=LENGTH_PENALTY_ALPHA
    )

    greedy_bleu_time = (
        time.time()
        - greedy_bleu_start
    ) / 60

    print(
        f"Greedy BLEU: "
        f"{greedy_bleu:.2f}"
    )

    print(
        f"Greedy BLEU time: "
        f"{greedy_bleu_time:.2f} min"
    )


    # --------------------------------------------------------
    # 23. Beam BLEU
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEAM BLEU")
    print("=" * 70)

    beam_bleu_start = time.time()

    beam_bleu = evaluate_bleu_bpe(
        model=model,
        dataset_subset=val_dataset,
        tokenizer=tokenizer,
        translate_fn=beam_search_decode_bpe,
        method="beam",
        max_examples=BLEU_MAX_EXAMPLES,
        max_len=MAX_LEN,
        beam_size=BEAM_SIZE,
        alpha=LENGTH_PENALTY_ALPHA
    )

    beam_bleu_time = (
        time.time()
        - beam_bleu_start
    ) / 60

    print(
        f"Beam BLEU: "
        f"{beam_bleu:.2f}"
    )

    print(
        f"Beam BLEU time: "
        f"{beam_bleu_time:.2f} min"
    )


    # --------------------------------------------------------
    # 24. Experiment configuration
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
            BATCH_SIZE,

        "initialization": INITIALIZATION,
        "lr_schedule": LR_SCHEDULE,
        "warmup_steps": WARMUP_STEPS,
        "label_smoothing": LABEL_SMOOTHING,
         "weight_tying": WEIGHT_TYING,
         "adam_beta1": ADAM_BETA1,
         "adam_beta2": ADAM_BETA2,
        "adam_eps": ADAM_EPS
    }


    # --------------------------------------------------------
    # 25. Save experiment summary
    # --------------------------------------------------------

    save_experiment_summary(
        file_path=summary_path,
        experiment_name=EXPERIMENT_NAME,
        config=experiment_config,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_perplexity=best_perplexity,
        total_time_min=total_time_min,
        checkpoint_path=CHECKPOINT_PATH,
        greedy_bleu=greedy_bleu,
        beam_bleu=beam_bleu
    )


    # --------------------------------------------------------
    # 26. Generate plots
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_paths = plot_training_history(
        history_csv_path=history_path,
        output_dir=PLOT_DIR,
        experiment_name=EXPERIMENT_NAME
    )


    # --------------------------------------------------------
    # 27. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
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
        "Greedy BLEU:",
        round(
            greedy_bleu,
            2
        )
    )

    print(
        "Beam BLEU:",
        round(
            beam_bleu,
            2
        )
    )

    print(
        "Training time:",
        round(
            total_time_min,
            2
        ),
        "minutes"
    )

    print(
        "Greedy BLEU time:",
        round(
            greedy_bleu_time,
            2
        ),
        "minutes"
    )

    print(
        "Beam BLEU time:",
        round(
            beam_bleu_time,
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
        "\nHistory:"
    )

    print(
        history_path
    )


    print(
        "\nResults:"
    )

    print(
        summary_path
    )


    print(
        "\nLoss plot:"
    )

    print(
        plot_paths[
            "loss_plot"
        ]
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
        plot_paths[
            "time_plot"
        ]
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()