import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional
from einops import rearrange, einsum
from minigpt.model.abstract_decoder import AbstractDecoder
from minigpt.config import GPTConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.emb_dim, 3 * config.emb_dim, bias=config.qkv_bias
        )
        # output projection
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_heads
        self.emb_dim = config.emb_dim
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # makes this tensor a persistent part of the model but not a parameter (it won't be updated during training)
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor):
        # batch size, sequence length, embedding dimensionality (emb_dim)
        b, s, emb_dim = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # batch size, sequence length, embedding dimensionality (emb_dim)
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=2)

        # d: head_dim = emb_dim // self.n_head
        # h: n_heads
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_head)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_head)

        # causal self-attention; Self-attend: (b, h, s, d) x (b, h, d, s) -> (b, h, s, s)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = einsum("b h s d, b h t d -> b h s t", q, k) * (
                1.0 / math.sqrt(k.size(-1))
            )
            # (b, h, s, s)
            att = att.masked_fill(self.bias[:, :, :s, :s] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = torch.einsum("b h s t, b h t d -> b h s d", att, v)
        y = rearrange(y, "b h s d -> b s (h d)")
        # h * d = emb dim
        y = self.resid_dropout(self.c_proj(y))
        return y


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            # 0.5 * x * (1+ torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
            nn.GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(AbstractDecoder):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.dropout)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.tok_emb.weight = self.out_head.weight

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        if labels is not None:
            # if we are given some desired label also calculate the loss
            logits = self.out_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.out_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length :]
            )

            # forward the model to get the logits for the index in the sequence
            logits, _ = self.forward(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
