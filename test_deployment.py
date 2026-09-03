from app.model_service import TranslationService


CHECKPOINT_PATH = (
    r"C:\Users\rudra\OneDrive\Transformer\checkpoints\bpe8000_d128_h8_enc4_dec4_ff512_bs32_lr0.00075"
    r"initdefaultschedconstanttieFalseb10.9_b20.98_eps1e-09.pt"
)

TOKENIZER_PATH = (
    r"C:\Users\rudra\OneDrive\Transformer\tokenizers\bpe_shared_8k.model"
)


service = TranslationService(
    checkpoint_path=CHECKPOINT_PATH,
    tokenizer_path=TOKENIZER_PATH,
)


german_sentence = (
    "Es war ganz unmöglich, "
    "an diesem Tage einen Spaziergang zu machen."
)


translation = service.translate(
    german_sentence
)


print("German:")
print(german_sentence)

print("\nEnglish:")
print(translation)