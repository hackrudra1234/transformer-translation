# inference/translate.py

from data.data_utils import encode_sentence
from inference.greedy import greedy_decode
from inference.beam import beam_search_decode


def translate_sentence(
    model,
    german_sentence,
    german_vocab,
    english_vocab,
    method="greedy",
    max_len=40,
    beam_size=4,
    alpha=0.6
):

    # --------------------------------
    # 1. German sentence -> token IDs
    # --------------------------------
    src = encode_sentence(
        german_sentence,
        german_vocab
    )

    # --------------------------------
    # 2. Choose decoding method
    # --------------------------------
    if method == "greedy":

        generated_ids = greedy_decode(
            model=model,
            src=src,
            german_vocab=german_vocab,
            english_vocab=english_vocab,
            max_len=max_len
        )

    elif method == "beam":

        generated_ids = beam_search_decode(
            model=model,
            src=src,
            german_vocab=german_vocab,
            english_vocab=english_vocab,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha
        )

    else:
        raise ValueError(
            "method must be 'greedy' or 'beam'"
        )

    # --------------------------------
    # 3. Reverse vocabulary
    # --------------------------------
    id_to_english = {
        idx: token
        for token, idx in english_vocab.items()
    }

    # --------------------------------
    # 4. IDs -> tokens
    # --------------------------------
    predicted_tokens = [
        id_to_english[idx.item()]
        for idx in generated_ids[0]
    ]

    # --------------------------------
    # 5. Remove special tokens
    # --------------------------------
    predicted_tokens = [
        token
        for token in predicted_tokens
        if token not in {
            "<bos>",
            "<eos>",
            "<pad>"
        }
    ]

    # --------------------------------
    # 6. Tokens -> sentence
    # --------------------------------
    predicted_sentence = " ".join(
        predicted_tokens
    )

    return predicted_sentence