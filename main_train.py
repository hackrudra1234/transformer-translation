# main_train.py

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
    CHECKPOINT_PATH
)

from data.data_utils import (
    build_vocab,
    create_collate_fn
)

from model.transformer import Transformer

from training.train import train_one_epoch
from training.evaluate import evaluate_model
from training.metrics import calculate_perplexity


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    # --------------------------------
    # 1. Reproducibility
    # --------------------------------
    set_seed(SEED)


    # --------------------------------
    # 2. Device
    # --------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------
    # 3. Load dataset
    # --------------------------------
    dataset = load_dataset(
        "Helsinki-NLP/opus_books",
        "de-en"
    )

    full_dataset = dataset["train"]

    print(
        "Total examples:",
        len(full_dataset)
    )


    # --------------------------------
    # 4. Train / validation split
    # --------------------------------
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


    # --------------------------------
    # 5. Build vocabulary
    # --------------------------------
    german_vocab,english_vocab = build_vocab(
        train_dataset,
        min_frequency=MIN_FREQUENCY
    )

    print(
        "German vocab:",
        len(german_vocab)
    )

    print(
        "English vocab:",
        len(english_vocab)
    )


    # --------------------------------
    # 6. Collate function
    # --------------------------------
    collate_fn = create_collate_fn(
        german_vocab,
        english_vocab
    )


    # --------------------------------
    # 7. DataLoaders
    # --------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )


    # --------------------------------
    # 8. Model
    # --------------------------------
    model = Transformer(
        src_vocab_size=len(german_vocab),
        tgt_vocab_size=len(english_vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


    # --------------------------------
    # 9. Loss
    # --------------------------------
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=english_vocab["<pad>"]
    )


    # --------------------------------
    # 10. Optimizer
    # --------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------
    # 11. Training
    # --------------------------------
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    val_perplexities = []

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        val_loss = evaluate_model(
            model,
            val_loader,
            loss_fn,
            device
        )

        perplexity = calculate_perplexity(
            val_loss
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        val_perplexities.append(
            perplexity
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PPL: {perplexity:.2f}"
        )


        # --------------------------------
        # 12. Save best checkpoint
        # --------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "german_vocab":
                        german_vocab,

                    "english_vocab":
                        english_vocab,

                    "config": {
                        "d_model": D_MODEL,
                        "num_heads": NUM_HEADS,
                        "d_ff": D_FF,
                        "num_encoder_layers":
                            NUM_ENCODER_LAYERS,
                        "num_decoder_layers":
                            NUM_DECODER_LAYERS,
                        "dropout": DROPOUT
                    },

                    "val_loss":
                        val_loss,

                    "perplexity":
                        perplexity
                },

                CHECKPOINT_PATH
            )

            print(
                "Best model saved:",
                CHECKPOINT_PATH
            )


    print("\nTraining complete.")

    print(
        "Best validation loss:",
        best_val_loss
    )

if __name__ == "__main__":
    main()