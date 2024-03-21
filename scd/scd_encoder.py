import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange

from model.ctrgcn import Model


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer) -> None:
        super().__init__()
        self.d_model = hidden_size

        hidden_size = 64
        self.gcn_t = Model(hidden_size)
        self.gcn_s = Model(hidden_size)

        self.channel_t = nn.Sequential(
            nn.Linear(50*hidden_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )

        self.channel_s = nn.Sequential(
            nn.Linear(64 * hidden_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )

        encoder_layer = TransformerEncoderLayer(self.d_model, num_head, self.d_model, batch_first=True)
        self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
        self.s_encoder = TransformerEncoder(encoder_layer, num_layer)

    def forward(self, x):
        
        vt = self.gcn_t(x)

        vt = rearrange(vt, '(B M) C T V -> B T (M V C)', M=2)
        vt = self.channel_t(vt)

        vs = self.gcn_s(x)
        
        vs = rearrange(vs, '(B M) C T V -> B (M V) (T C)', M=2)
        vs = self.channel_s(vs)

        vt = self.t_encoder(vt) # B T C

        vs = self.s_encoder(vs)

        # implementation using amax for the TMP runs faster than using MaxPool1D
        # not support pytorch < 1.7.0
        vt = vt.amax(dim=1)
        vs = vs.amax(dim=1)

        return vt, vs


class PretrainingEncoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer,
                 num_class=60,
                 ):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size

        self.encoder = Encoder(
            hidden_size, num_head, num_layer,
        )

        # temporal feature projector
        self.t_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # spatial feature projector
        self.s_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # instance level feature projector
        self.i_proj = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, x):

        vt, vs = self.encoder(x)

        # projection
        zt = self.t_proj(vt)
        zs = self.s_proj(vs)

        vi = torch.cat([vt, vs], dim=1)

        zi = self.i_proj(vi)

        return zt, zs, zi


class DownstreamEncoder(nn.Module):
    """hierarchical encoder network + classifier"""

    def __init__(self, 
                 hidden_size, num_head, num_layer,
                 num_class=60,
                 ):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.encoder = Encoder(
            hidden_size, num_head, num_layer,
        )

        # linear classifier
        self.fc = nn.Linear(2 * self.d_model, num_class)

    def forward(self, x, knn_eval=False):

        vt, vs = self.encoder(x)

        vi = torch.cat([vt, vs], dim=1)

        if knn_eval:  # return last layer features during  KNN evaluation (action retrieval)
            return vi
        else:
            return self.fc(vi)