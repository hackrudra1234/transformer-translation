# model/encoder.py

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

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

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(
        self,
        x,
        src_mask=None
    ):

        # --------------------------------
        # 1. Encoder self-attention
        # --------------------------------
        attention_output, attention_weights = self.self_attention(
            x,
            x,
            x,
            mask=src_mask
        )

        # --------------------------------
        # 2. Residual + LayerNorm
        # --------------------------------
        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        # --------------------------------
        # 3. Feed Forward
        # --------------------------------
        ff_output = self.ffn(x)

        # --------------------------------
        # 4. Residual + LayerNorm
        # --------------------------------
        x = self.norm2(
            x + self.dropout2(ff_output)
        )

        return x, attention_weights

    
class Encoder(nn.Module):

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
            EncoderLayer(
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
        src_mask=None
    ):

        attention_maps = []

        for layer in self.layers:

            x, attention_weights = layer(
                x,
                src_mask
            )

            attention_maps.append(
                attention_weights
            )

        return x, attention_maps