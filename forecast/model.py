import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, feature_size, nhead, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(feature_size, nhead, dropout=dropout)
        self.linear1 = nn.Linear(feature_size, feature_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feature_size * 4, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, feature_size, nhead, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(feature_size, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            feature_size, nhead, dropout=dropout
        )
        self.linear1 = nn.Linear(feature_size, feature_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feature_size * 4, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.norm3 = nn.LayerNorm(feature_size)

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(
        self, feature_size=7, num_layers=3, nhead=7, dropout=0.1, device="cpu"
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(feature_size, nhead, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(feature_size, nhead, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src):
        src = src.to(self.device)

        src_mask = self._generate_square_subsequent_mask(src.size(0)).to(self.device)
        tgt_mask = self._generate_square_subsequent_mask(src.size(0)).to(self.device)

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory)

        output = memory
        for layer in self.decoder_layers:
            output = layer(output, memory)

        output = self.decoder(output)
        return output
