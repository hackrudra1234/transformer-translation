# training/evaluate.py

import torch


def evaluate_model(
    model,
    val_loader,
    loss_fn,
    device
):

    # -----------------------------
    # Evaluation mode
    # -----------------------------
    model.eval()

    total_loss = 0.0

    # -----------------------------
    # No gradients needed
    # -----------------------------
    with torch.no_grad():

        for batch in val_loader:

            # -----------------------------
            # Move batch to device
            # -----------------------------
            src = batch["src"].to(
                device,
                non_blocking=True
            )

            decoder_input = batch["decoder_input"].to(
                device,
                non_blocking=True
            )

            target = batch["target"].to(
                device,
                non_blocking=True
            )

            src_mask = batch["src_mask"].to(
                device,
                non_blocking=True
            )

            tgt_mask = batch["tgt_mask"].to(
                device,
                non_blocking=True
            )

            # -----------------------------
            # Forward pass
            # -----------------------------
            logits, _, _, _ = model(
                src,
                decoder_input,
                src_mask,
                tgt_mask
            )

            # -----------------------------
            # Validation loss
            # -----------------------------
            loss = loss_fn(
                logits.reshape(
                    -1,
                    logits.shape[-1]
                ),
                target.reshape(-1)
            )

            total_loss += loss.item()

    average_loss = (
        total_loss
        / len(val_loader)
    )

    return average_loss