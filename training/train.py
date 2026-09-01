# training/train.py

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    scheduler=None
):

    model.train()

    total_loss = 0.0

    for batch in train_loader:

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
        # Clear old gradients
        # -----------------------------
        optimizer.zero_grad()


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
        # Loss
        # -----------------------------
        loss = loss_fn(
            logits.reshape(
                -1,
                logits.shape[-1]
            ),
            target.reshape(-1)
        )


        # -----------------------------
        # Backpropagation
        # -----------------------------
        loss.backward()


        # -----------------------------
        # Update parameters
        # -----------------------------
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()


    average_loss = (
        total_loss
        / len(train_loader)
    )

    return average_loss