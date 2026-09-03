# model/transformer.py

import torch.nn as nn

from model.embeddings import Embeddings
from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_encoder_layers,
        num_decoder_layers,
        dropout=0.1,
        weight_tying=False
    ):
        super().__init__()

        # -----------------------------
        # Source embedding
        # -----------------------------
        self.src_embedding = Embeddings(
            vocab_size=src_vocab_size,
            d_model=d_model
        )

        # -----------------------------
        # Target embedding
        # -----------------------------
        self.tgt_embedding = Embeddings(
            vocab_size=tgt_vocab_size,
            d_model=d_model
        )

        # -----------------------------
        # Positional encoding
        # -----------------------------
        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        # -----------------------------
        # Encoder
        # -----------------------------
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # -----------------------------
        # Decoder
        # -----------------------------
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # -----------------------------
        # Final vocabulary projection
        # -----------------------------
        self.output_linear = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        if weight_tying:

            self.output_linear.weight = (
                self.tgt_embedding.embedding.weight
            )
    # =========================================================
    # ENCODER
    # =========================================================

    def encode(self, src, src_mask=None):
        src_x = self.src_embedding(src)

        src_x = self.position_encoding(
            src_x
        )

        (encoder_output, encoder_attention) = self.encoder(
            src_x,
            src_mask
        )

        return encoder_output, encoder_attention
    # =========================================================
    # DECODER
    # =========================================================

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        tgt_x = self.tgt_embedding(tgt)

        tgt_x = self.position_encoding(
            tgt_x
        )

        (
            decoder_output,
            decoder_self_attention,
            decoder_cross_attention
        ) = self.decoder(
            tgt_x,
            encoder_output,
            tgt_mask,
            src_mask
        )
        logits = self.output_linear(decoder_output)


        return (
            logits,
            decoder_self_attention,
            decoder_cross_attention
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):


        (encoder_output, encoder_attention) = self.encode(src, src_mask)

        (logits, decoder_self_attention, decoder_cross_attention) = self.decode(
            tgt,encoder_output, tgt_mask, src_mask)

        return (logits, encoder_attention, decoder_self_attention, decoder_cross_attention
        )