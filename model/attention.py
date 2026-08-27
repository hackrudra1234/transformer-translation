# model/attention.py

import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_k = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_v = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_o = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


    def forward(
        self,
        Q,
        K,
        V,
        mask=None
    ):

        batch_size = Q.shape[0]

        seq_len_q = Q.shape[1]
        seq_len_k = K.shape[1]

        # --------------------------------
        # 1. Linear projections
        # --------------------------------
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # --------------------------------
        # 2. Split into heads
        # --------------------------------
        Q = Q.view(
            batch_size,
            seq_len_q,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len_k,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len_k,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        # Shapes:
        # Q = [batch, heads, query_len, d_k]
        # K = [batch, heads, key_len, d_k]
        # V = [batch, heads, key_len, d_k]

        # --------------------------------
        # 3. Scaled dot-product scores
        # --------------------------------
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = (
            scores
            / math.sqrt(self.d_k)
        )

        # scores:
        # [batch, heads, query_len, key_len]

        # --------------------------------
        # 4. Apply mask
        # --------------------------------
        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                -1e9
            )

        # --------------------------------
        # 5. Attention probabilities
        # --------------------------------
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # 6. Weighted sum of V
        # --------------------------------
        head_outputs = torch.matmul(
            attention_weights,
            V
        )

        # [batch, heads, query_len, d_k]

        # --------------------------------
        # 7. Join heads
        # --------------------------------
        head_outputs = (
            head_outputs
            .transpose(1, 2)
            .contiguous()
        )

        concat = head_outputs.view(
            batch_size,
            seq_len_q,
            self.d_model
        )

        # --------------------------------
        # 8. Final projection
        # --------------------------------
        output = self.W_o(concat)

        return (
            output,
            attention_weights
        )