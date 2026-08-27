# inference/greedy.py

import torch


def greedy_decode(
    model,
    src,
    german_vocab,
    english_vocab,
    max_len=40
):

    model.eval()

    device = next(model.parameters()).device

    bos_id = english_vocab["<bos>"]
    eos_id = english_vocab["<eos>"]
    pad_id = english_vocab["<pad>"]

    # --------------------------------
    # Source to device
    # --------------------------------
    src = src.to(device)

    # If one sentence has shape:
    # [src_len]
    # convert to:
    # [1, src_len]
    if src.dim() == 1:
        src = src.unsqueeze(0)

    # --------------------------------
    # Source mask
    # --------------------------------
    src_mask = (
        src != german_vocab["<pad>"]
    ).unsqueeze(1).unsqueeze(2)

    # --------------------------------
    # Start decoder with <bos>
    # --------------------------------
    generated = torch.tensor(
        [[bos_id]],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_len):

            tgt_len = generated.shape[1]

            # --------------------------------
            # Causal mask
            # --------------------------------
            causal_mask = torch.tril(
                torch.ones(
                    tgt_len,
                    tgt_len,
                    dtype=torch.bool,
                    device=device
                )
            )

            causal_mask = (
                causal_mask
                .unsqueeze(0)
                .unsqueeze(1)
            )

            # --------------------------------
            # Target padding mask
            # --------------------------------
            tgt_padding_mask = (
                generated != pad_id
            ).unsqueeze(1).unsqueeze(2)

            # --------------------------------
            # Final target mask
            # --------------------------------
            tgt_mask = (
                tgt_padding_mask
                &
                causal_mask
            )

            # --------------------------------
            # Transformer forward pass
            # --------------------------------
            logits, _, _, _ = model(
                src,
                generated,
                src_mask,
                tgt_mask
            )

            # Last decoder position only
            next_token_logits = logits[:, -1, :]

            # Greedy choice
            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1
            )

            # Append predicted token
            generated = torch.cat(
                [
                    generated,
                    next_token_id.unsqueeze(1)
                ],
                dim=1
            )

            # Stop on EOS
            if next_token_id.item() == eos_id:
                break

    return generated