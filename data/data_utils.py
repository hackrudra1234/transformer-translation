import re
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def simple_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def build_vocab (
    dataset_split,
    min_frequency = 2
):
    # -----------------------------
    # 1. Count tokens
    # -----------------------------
    german_counter = Counter()
    english_counter = Counter()

    for example in dataset_split:

        german_sentence = (example["translation"]["de"])
        english_sentence = (example["translation"]["en"])

        german_tokens = simple_tokenize(
            german_sentence.lower()
        )

        english_tokens = simple_tokenize(
            english_sentence.lower()
        )

        german_counter.update(german_tokens)
        english_counter.update(english_tokens)


    # -----------------------------
    # 2. Special tokens
    # -----------------------------
    special_tokens = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    ]


    # -----------------------------
    # 3. German vocabulary
    # -----------------------------
    german_vocab = {}

    for token in special_tokens:
        german_vocab[token] = len(german_vocab)

    for token, count in german_counter.items():
        if count >= min_frequency:
            german_vocab[token] = len(german_vocab)


    # -----------------------------
    # 4. English vocabulary
    # -----------------------------
    english_vocab = {}

    for token in special_tokens:
        english_vocab[token] = len(english_vocab)

    for token, count in english_counter.items():
        if count >= min_frequency:
            english_vocab[token] = len(english_vocab)
    return (
        german_vocab,
        english_vocab

    )

def encode_sentence(
    sentence,
    vocab
):
    tokens = simple_tokenize(
        sentence.lower()
    )

    ids = [
        vocab["<bos>"]
    ]

    ids += [
        vocab.get(
            token,
            vocab["<unk>"]
        )
        for token in tokens
    ]

    ids.append(
        vocab["<eos>"]
    )

    return torch.tensor(
        ids,
        dtype=torch.long
    )

def create_collate_fn(
    german_vocab,
    english_vocab
):

    def collate_fn(batch):

        src_sequences = []
        decoder_sequences = []
        target_sequences = []

        for example in batch:

            src = encode_sentence(
                example["translation"]["de"],
                german_vocab
            )

            tgt = encode_sentence(
                example["translation"]["en"],
                english_vocab
            )

            src_sequences.append(src)

            decoder_sequences.append(
                tgt[:-1]
            )

            target_sequences.append(
                tgt[1:]
            )

        # -------------------------
        # Padding
        # -------------------------

        src_batch = pad_sequence(
            src_sequences,
            batch_first=True,
            padding_value=german_vocab["<pad>"]
        )

        decoder_batch = pad_sequence(
            decoder_sequences,
            batch_first=True,
            padding_value=english_vocab["<pad>"]
        )

        target_batch = pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=english_vocab["<pad>"]
        )

        # -------------------------
        # Source mask
        # -------------------------

        src_mask = (
            src_batch != german_vocab["<pad>"]
        ).unsqueeze(1).unsqueeze(2)

        # -------------------------
        # Decoder padding mask
        # -------------------------

        tgt_padding_mask = (
            decoder_batch != english_vocab["<pad>"]
        ).unsqueeze(1).unsqueeze(2)

        # -------------------------
        # Causal mask
        # -------------------------

        tgt_len = decoder_batch.shape[1]

        causal_mask = torch.tril(
            torch.ones(
                tgt_len,
                tgt_len,
                dtype=torch.bool
            )
        )

        causal_mask = (
            causal_mask
            .unsqueeze(0)
            .unsqueeze(1)
        )

        # padding + causal
        tgt_mask = (
            tgt_padding_mask
            &
            causal_mask
        )

        return {
            "src": src_batch,
            "decoder_input": decoder_batch,
            "target": target_batch,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask
        }

    return collate_fn