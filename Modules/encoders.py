import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from .normalizations import *
from .attentions import Attention
from .embeddings import NumberEmbedder
#from resblock import ResBlk
from Utility.utils import length_to_mask
from torchaudio.models import Conformer
import math

class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class StyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6):
        super().__init__()        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.conformer_pre = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.out = nn.Linear(text_dim, style_dim)
        
    def forward(self, mel, text, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel = self.mel_proj(mel)
        mel_len = mel.size(-1) # length of mel

        input_lengths = input_lengths + mel_len
        x = torch.cat([mel, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, output_lengths = self.conformer_pre(x, input_lengths)
        x, output_lengths = self.conformer_body(x, input_lengths)
        x = x.transpose(-1, -2)
        x_mel = x[:, :, :mel_len]
        x_text = x[:, :, mel_len:]
        
        s = self.out(x_mel.mean(axis=-1))
        text_return[:, :, :x_text.size(-1)] = x_text
        
        
        return s, text_return


class TVStyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, text_dim=512, 
                 num_heads=8, num_time=50, num_layers=6,
                 head_features=64):
        super().__init__()
        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        
        self.conformer_pre = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        
        max_conv_dim = text_dim
        
        self.cross_attention = Attention(
            features=max_conv_dim,
            num_heads=num_heads,
            head_features=head_features,
            context_features=max_conv_dim,
            use_rel_pos=False
        )
        self.num_time = num_time
        self.positions = nn.Embedding(num_time, max_conv_dim)
        
        self.embedder = NumberEmbedder(features=max_conv_dim)
        
    def forward(self, x, input_lengths):
        x = x[..., :input_lengths.max()]
        
        x = self.mel_proj(x)
        x = x.transpose(-1, -2)
        x, output_lengths = self.conformer_pre(x, input_lengths)
        x, output_lengths = self.conformer_body(x, input_lengths)
        h = x.transpose(-1, -2)
        
        idx = torch.arange(0, self.num_time).to(x.device)
        positions = self.positions(idx).transpose(-1, -2).expand(x.shape[0], -1, -1)
        
        pos = self.embedder(torch.arange(h.shape[-1]).to(x.device)).transpose(-1, -2).expand(h.size(0), -1, -1)
        h += pos
                
        m = length_to_mask(input_lengths).to(x.device)
        h.masked_fill_(m.unsqueeze(1), 0.0)
        h = self.cross_attention(positions.transpose(-1, -2), context=h.transpose(-1, -2))
        
        return h.transpose(-1, -2)


class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        
        return x.transpose(-1, -2)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask