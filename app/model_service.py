import torch

from model.transformer import Transformer
from data.bpe_tokenizers import load_bpe_tokenizer
from inference.beam import beam_search_decode_bpe

class TranslationService:
    def __init__(self,checkpoint_path, tokenizer_path, device=None,max_len=80, beam_size=4,alpha=1.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.beam_size = beam_size
        self.alpha = alpha

        # 1.Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)

        # 2.Load tokenizer
        #bpe_model_path = checkpoint['bpe_model_path']
        self.tokenizer = load_bpe_tokenizer(tokenizer_path)
        vocab_size = checkpoint['bpe_vocab_size']


        # 3.Read saved model
        config = checkpoint['config']

        # 4. Recreate exact Transformer architecture
        self.model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dropout=config['dropout'],
            weight_tying=config['weight_tying'],
        )

        # 5. Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # 6. Deployment/Inference mode
        self.model.eval()

        print("Translation model loaded")
        print("Device:", self.device)
        print("BPE vocab:", vocab_size)
        print("Best checkpoint epoch:",checkpoint["epoch"])

    def translate(self, sentence):
            if not sentence or not sentence.strip():
                raise ValueError("Input sentence is empty or None.")
            with torch.inference_mode():
                translation = beam_search_decode_bpe(model = self.model,
                                                     sentence=sentence,
                                                     tokenizer=self.tokenizer,
                                                        device=self.device,
                                                        max_len=self.max_len,
                                                        beam_size=self.beam_size,
                                                        alpha=self.alpha)
            return translation
                                                