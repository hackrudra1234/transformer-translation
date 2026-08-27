# main_inference.py

import torch

from config import (
    CHECKPOINT_PATH,
    MAX_LEN,
    BEAM_SIZE,
    LENGTH_PENALTY_ALPHA
)

from model.transformer import Transformer
from inference.translate import translate_sentence


def main():

    # --------------------------------
    # 1. Device
    # --------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    # --------------------------------
    # 2. Load checkpoint
    # --------------------------------
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )


    # --------------------------------
    # 3. Load vocabularies
    # --------------------------------
    german_vocab = checkpoint[
        "german_vocab"
    ]

    english_vocab = checkpoint[
        "english_vocab"
    ]


    # --------------------------------
    # 4. Load saved model config
    # --------------------------------
    model_config = checkpoint[
        "config"
    ]


    # --------------------------------
    # 5. Recreate model architecture
    # --------------------------------
    model = Transformer(
        src_vocab_size=len(german_vocab),
        tgt_vocab_size=len(english_vocab),

        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],

        num_encoder_layers=
            model_config["num_encoder_layers"],

        num_decoder_layers=
            model_config["num_decoder_layers"],

        dropout=model_config["dropout"]
    ).to(device)


    # --------------------------------
    # 6. Load trained weights
    # --------------------------------
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation loss:",
        checkpoint["val_loss"]
    )

    print(
        "Perplexity:",
        checkpoint["perplexity"]
    )


    # --------------------------------
    # 7. German test sentence
    # --------------------------------
    german_sentence = (
        "Es war ein schöner Tag."
    )


    # --------------------------------
    # 8. Greedy translation
    # --------------------------------
    greedy_translation = translate_sentence(
        model=model,
        german_sentence=german_sentence,
        german_vocab=german_vocab,
        english_vocab=english_vocab,
        method="greedy",
        max_len=MAX_LEN
    )


    # --------------------------------
    # 9. Beam translation
    # --------------------------------
    beam_translation = translate_sentence(
        model=model,
        german_sentence=german_sentence,
        german_vocab=german_vocab,
        english_vocab=english_vocab,
        method="beam",
        max_len=MAX_LEN,
        beam_size=BEAM_SIZE,
        alpha=LENGTH_PENALTY_ALPHA
    )


    # --------------------------------
    # 10. Results
    # --------------------------------
    print("\nGerman:")
    print(german_sentence)

    print("\nGreedy:")
    print(greedy_translation)

    print("\nBeam:")
    print(beam_translation)


if __name__ == "__main__":
    main()