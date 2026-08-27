# training/metrics.py

import math
import sacrebleu


def calculate_perplexity(loss):
    return math.exp(
        min(loss, 20)
    )


def evaluate_bleu(
    model,
    dataset_subset,
    german_vocab,
    english_vocab,
    translate_fn,
    method="greedy",
    max_examples=50,
    max_len=40,
    beam_size=4,
    alpha=0.6
):

    predictions = []
    references = []

    n = min(
        max_examples,
        len(dataset_subset)
    )

    for i in range(n):

        example = dataset_subset[i]["translation"]

        german_sentence = example["de"]
        actual_english = example["en"]

        predicted_english = translate_fn(
            model=model,
            german_sentence=german_sentence,
            german_vocab=german_vocab,
            english_vocab=english_vocab,
            method=method,
            max_len=max_len,
            beam_size=beam_size,
            alpha=alpha
        )

        predictions.append(
            predicted_english
        )

        references.append(
            actual_english
        )

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references]
    )

    return bleu.score