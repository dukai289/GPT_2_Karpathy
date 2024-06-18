# -*- coding: utf-8 -*-

from dataclass import dataclass
from torch import nn
import torch
from torch.nn import functiional as F

@dataclass
class TrainConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class CasualSelfAttention(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        ...

class MLP(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        ...

class Block(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed)
           ,wpe = nn.Embedding(config.block_size, config.n_embed)
           ,h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
           ,ln_f = nn.LayerNorm(config.n_embed))
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        ...

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape(t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape(t, n_embed)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1), targets.view(-1), ignore_index=-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        # there are performace reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        ...