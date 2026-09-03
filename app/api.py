import os
from fastapi import FastAPI
from pydantic import BaseModel

from app.model_service import TranslationService

app = FastAPI(title = "German to English Translation")

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    r"C:\Users\rudra\OneDrive\Transformer\checkpoints\bpe8000_d128_h8_enc4_dec4_ff512_bs32_lr0.00075initdefaultschedconstanttieFalseb10.9_b20.98_eps1e-09.pt"
)

TOKENIZER_PATH = os.getenv(
    "TOKENIZER_PATH",
    r"C:\Users\rudra\OneDrive\Transformer\tokenizers\bpe_shared_8k.model"
)


print("CHECKPOINT_PATH:", CHECKPOINT_PATH)
print("TOKENIZER_PATH:", TOKENIZER_PATH)



service = TranslationService(CHECKPOINT_PATH, TOKENIZER_PATH)

class TranslationRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the German to English Translation API!"}

@app.post("/translate")
def translate(request: TranslationRequest):
    translation = service.translate(request.text)
    return {"source": request.text, "translation": translation}