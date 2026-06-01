"""Sequence models: LSTM, GRU, and a small Transformer encoder.

All three take:
  x          : (B, W, F)  the input window of W timesteps with F features
  station_id : (B,)       which of the 12 stations we're predicting for

and produce a single scalar per example -- the (standardized) PM2.5 value
at the target timestep.

A small learned station embedding is concatenated to the per-timestep
feature vector before the sequence backbone. That way one global model
can condition on station identity without having to memorize a discrete
id inside its recurrent / attention layers.
"""
import math

import torch
import torch.nn as nn

from . import config


class StationEmbedding(nn.Module):
    """Look up an 8-dim embedding per station and broadcast it across the
    time dimension so it can be concatenated to the per-timestep features."""

    def __init__(self, n_stations, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(n_stations, embed_dim)

    def forward(self, station_id, seq_len):
        e = self.embed(station_id)              # (B, embed_dim)
        return e.unsqueeze(1).expand(-1, seq_len, -1)   # (B, seq_len, embed_dim)


class LSTMRegressor(nn.Module):
    """2-layer LSTM -> last hidden state -> small MLP -> scalar."""

    def __init__(self, input_dim, hidden=128, num_layers=2, dropout=0.2,
                 n_stations=config.N_STATIONS, station_embed=8):
        super().__init__()
        self.station = StationEmbedding(n_stations, station_embed)
        self.lstm = nn.LSTM(
            input_dim + station_embed, hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, station_id):
        e = self.station(station_id, x.size(1))
        h, _ = self.lstm(torch.cat([x, e], dim=-1))
        return self.head(h[:, -1]).squeeze(-1)


class GRURegressor(nn.Module):
    """Same as LSTMRegressor but with GRU cells."""

    def __init__(self, input_dim, hidden=128, num_layers=2, dropout=0.2,
                 n_stations=config.N_STATIONS, station_embed=8):
        super().__init__()
        self.station = StationEmbedding(n_stations, station_embed)
        self.gru = nn.GRU(
            input_dim + station_embed, hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, station_id):
        e = self.station(station_id, x.size(1))
        h, _ = self.gru(torch.cat([x, e], dim=-1))
        return self.head(h[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from the Transformer paper."""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    """Small Transformer encoder. We keep it small (d_model=128, 2 layers,
    4 heads) to fit comfortably on MPS and to keep training cheap."""

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 dropout=0.1, n_stations=config.N_STATIONS, station_embed=8):
        super().__init__()
        self.station = StationEmbedding(n_stations, station_embed)
        self.input_proj = nn.Linear(input_dim + station_embed, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, station_id):
        e = self.station(station_id, x.size(1))
        z = self.input_proj(torch.cat([x, e], dim=-1))
        z = self.pos(z)
        z = self.encoder(z)
        return self.head(z[:, -1]).squeeze(-1)


def build_model(name, input_dim):
    """Tiny factory so the runner can pick a model by string name."""
    name = name.lower()
    if name == "lstm":
        return LSTMRegressor(input_dim)
    if name == "gru":
        return GRURegressor(input_dim)
    if name in ("transformer", "tfm"):
        return TransformerRegressor(input_dim)
    raise ValueError(f"unknown sequence model: {name!r}")


SEQUENCE_NAMES = ("gru", "lstm", "transformer")
