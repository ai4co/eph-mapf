import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid):
        """Initialization"""
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x):
        """run a ff layer"""
        x = self.w_2(F.relu(self.w_1(x)))
        return x



class EncoderLayer(nn.Module):
    """compose with two different sub-layers"""

    def __init__(self, d_model, d_hidden, n_head):
        """define one computation block"""
        super(EncoderLayer, self).__init__()
        # self.gate1 = GatingMechanism(d_model)
        # self.gate2 = GatingMechanism(d_model)
        # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.slf_attn = nn.MultiheadAttention(d_model, n_head)#, d_k, d_v)    
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hidden)
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))

    def forward(self, enc_input):
        """run a computation block"""
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output_1 = enc_input + self.alpha1 * enc_output # ReZero
        enc_output = self.pos_ffn(enc_output_1)
        enc_output_2 = enc_output_1 + self.alpha2 * enc_output # ReZero
        return enc_output_2, enc_slf_attn
    

class Encoder(nn.Module):
    """a encoder model with self attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head):
        """create multiple computation blocks"""
        super().__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_head) for _ in range(n_layers)])

    def forward(self, enc_output, return_attns=False):
        """use self attention to merge messages"""
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class PositionalEncoding(nn.Module):
    """sinusoidal position embedding"""

    def __init__(self, d_hid, n_position=200):
        """create table"""
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """encode unique agent id """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    """a sequence to sequence model with attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, n_position):
        """initialization"""
        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_hidden=d_hidden,
                               n_layers=n_layers, n_head=n_head)

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

    def forward(self, encoder_input):
        """run encoder"""
        encoder_input = self.position_enc(encoder_input)

        enc_output, *_ = self.encoder(encoder_input)

        return enc_output
    
    
if __name__ == "__main__":
    # test
    model = TransformerEncoder(128, 256, 2, 8, 200)
    input = torch.rand(32, 10, 128)
    output = model(input)
    print(output.size())