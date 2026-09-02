import os
import sentencepiece as spm


def train_bpe_tokenizer(
    train_dataset,
    model_prefix,
    vocab_size=16000
):
    """
    Train one BPE tokenizer using both German and English
    sentences from the TRAINING split only.
    """

    corpus_path = f"{model_prefix}_corpus.txt"

    # --------------------------------------------------
    # 1. Write training text to a temporary corpus file
    # --------------------------------------------------

    with open(corpus_path, "w", encoding="utf-8") as f:

        for example in train_dataset:

            german = example["translation"]["de"]
            english = example["translation"]["en"]

            f.write(german.strip() + "\n")
            f.write(english.strip() + "\n")

    # --------------------------------------------------
    # 2. Train SentencePiece BPE
    # --------------------------------------------------

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",

        character_coverage=1.0
    )

    # remove temporary text file
    os.remove(corpus_path)

    print("BPE tokenizer trained.")
    print("Model saved at:", model_prefix + ".model")


def load_bpe_tokenizer(model_path):
    """
    Load an already trained SentencePiece tokenizer.
    """

    tokenizer = spm.SentencePieceProcessor()

    tokenizer.load(model_path)

    return tokenizer