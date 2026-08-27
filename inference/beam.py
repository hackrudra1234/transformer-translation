# inference/beam.py

import torch


def beam_search_decode(
    model,
    src,
    german_vocab,
    english_vocab,
    beam_size=4,
    max_len=40,
    alpha=0.6
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

    # [src_len] -> [1, src_len]
    if src.dim() == 1:
        src = src.unsqueeze(0)

    # --------------------------------
    # Source mask
    # --------------------------------
    src_mask = (
        src != german_vocab["<pad>"]
    ).unsqueeze(1).unsqueeze(2)

    # --------------------------------
    # Initial beam:
    # sequence + cumulative log score
    # --------------------------------
    beams = [
        (
            torch.tensor(
                [[bos_id]],
                dtype=torch.long,
                device=device
            ),
            0.0
        )
    ]

    with torch.no_grad():

        for _ in range(max_len):

            candidates = []

            # Expand each current beam
            for generated, score in beams:

                # If this beam already finished,
                # keep it unchanged
                if generated[0, -1].item() == eos_id:

                    candidates.append(
                        (generated, score)
                    )

                    continue

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

                tgt_mask = (
                    tgt_padding_mask
                    &
                    causal_mask
                )

                # --------------------------------
                # Forward pass
                # --------------------------------
                logits, _, _, _ = model(
                    src,
                    generated,
                    src_mask,
                    tgt_mask
                )

                # Last decoder position
                next_token_logits = logits[:, -1, :]

                # Convert to log probabilities
                log_probs = torch.log_softmax(
                    next_token_logits,
                    dim=-1
                )

                # Best beam_size next tokens
                top_log_probs, top_ids = torch.topk(
                    log_probs,
                    beam_size,
                    dim=-1
                )

                # --------------------------------
                # Create candidate sequences
                # --------------------------------
                for i in range(beam_size):

                    next_token = top_ids[0, i]

                    new_score = (
                        score
                        +
                        top_log_probs[0, i].item()
                    )

                    new_generated = torch.cat(
                        [
                            generated,
                            next_token.view(1, 1)
                        ],
                        dim=1
                    )

                    candidates.append(
                        (
                            new_generated,
                            new_score
                        )
                    )

            # --------------------------------
            # Length-normalized score
            # --------------------------------
            def normalized_score(item):

                sequence, score = item

                length = sequence.shape[1]

                length_penalty = (
                    (5 + length) / 6
                ) ** alpha

                return score / length_penalty

            # --------------------------------
            # Keep only best beams
            # --------------------------------
            candidates = sorted(
                candidates,
                key=normalized_score,
                reverse=True
            )

            beams = candidates[:beam_size]

            # --------------------------------
            # Stop if all beams reached EOS
            # --------------------------------
            if all(
                sequence[0, -1].item() == eos_id
                for sequence, _ in beams
            ):
                break

    # --------------------------------
    # Return best final sequence
    # --------------------------------
    best_sequence, _ = max(
        beams,
        key=normalized_score
    )

    return best_sequence