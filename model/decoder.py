# model/decoder.py

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None
    ):

        # --------------------------------
        # 1. Masked decoder self-attention
        # --------------------------------
        self_attn_output, self_attn_weights = self.self_attention(
            x,
            x,
            x,
            mask=tgt_mask
        )

        x = self.norm1(
            x + self.dropout1(self_attn_output)
        )

        # --------------------------------
        # 2. Encoder-decoder cross-attention
        # --------------------------------
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            mask=src_mask
        )

        x = self.norm2(
            x + self.dropout2(cross_attn_output)
        )

        # --------------------------------
        # 3. Feed Forward
        # --------------------------------
        ff_output = self.ffn(x)

        x = self.norm3(
            x + self.dropout3(ff_output)
        )

        return (
            x,
            self_attn_weights,
            cross_attn_weights
        )
class Decoder(nn.Module):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])


    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None
    ):

        self_attention_maps = []
        cross_attention_maps = []

        for layer in self.layers:

            x, self_attn, cross_attn = layer(
                x,
                encoder_output,
                tgt_mask,
                src_mask
            )

            self_attention_maps.append(
                self_attn
            )

            cross_attention_maps.append(
                cross_attn
            )

        return (
            x,
            self_attention_maps,
            cross_attention_maps
        )