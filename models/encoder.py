from re import L
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from contextlib import contextmanager
from typing import Union
from models.attention import DecoderLayer, sinusoid_encoding_table, ShallowLayer, middle_layer

TensorOrNone = Union[torch.Tensor, None]


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.a_feature_size = opt.a_feature_size
        self.m_feature_size = opt.m_feature_size
        self.hidden_size = opt.hidden_size
        self.concat_size = self.a_feature_size + self.m_feature_size
        self.use_multi_gpu = opt.use_multi_gpu

        self.cnn_emd_Q = nn.Sequential(
            nn.Linear(self.a_feature_size + self.m_feature_size, self.hidden_size),
            nn.Dropout(p=0.2)
            )

        self.depth_emd_Q = nn.Sequential(
            nn.Linear(512, self.hidden_size),
            nn.Dropout(p=0.2)
            )

        # frame feature embedding
        self.frame_feature_embed = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.shallow_att = nn.ModuleList(
            [ShallowLayer(self.hidden_size, self.hidden_size // 8, self.hidden_size // 8, 8) for _ in range(1)])
        
        self.middle_att = nn.ModuleList(
            [middle_layer(self.hidden_size, self.hidden_size // 8, self.hidden_size // 8, 8) for _ in range(1)])

        # self.parallel_att = nn.ModuleList(
        #     [DecoderLayer(self.hidden_size, self.hidden_size // 8, self.hidden_size // 8, 8) for _ in range(8)])

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.shallow_lnorm_v = nn.LayerNorm(self.hidden_size)
        self.shallow_lnorm_d = nn.LayerNorm(self.hidden_size)
        
        self.middle_lnorm_v = nn.LayerNorm(self.hidden_size)
        self.middle_lnorm_d = nn.LayerNorm(self.hidden_size)

    def _init_weights(self):
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        nn.init.constant_(self.frame_feature_embed.bias, 0)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    # forward for test
    def forward(self, cnn_feats, depth_feats):
        '''
        :param cnn_feats: (batch_size, max_frames, m_feature_size + a_feature_size)
        :param depth_feats: (batch_sizem max_frames, hidden_size)
        :param region_feats: (batch_size, max_frames, num_boxes, region_feature_size)
        :param spatial_feats: (batch_size, max_frames, num_boxes, spatial_feature_size)
        :return: output of Bidirectional LSTM and embedded region features
        '''
        # 2d cnn or 3d cnn or 2d+3d cnn

        assert self.a_feature_size + self.m_feature_size == cnn_feats.size(2)
        # assert depth_feats.size(-1) == self.hidden_size 

        # visual_feats = self.cnn_emd_Q(torch.cat([cnn_feats, depth_feats], dim=-1))
        visual_feats = self.cnn_emd_Q(cnn_feats)
        depth_feats = self.depth_emd_Q(depth_feats)
    
        # #? depth model
        # #? early layer 
        for i, l in enumerate(self.shallow_att):
            visual_feats, depth_feats = l(visual_feats, depth_feats)

        visual_feats = self.shallow_lnorm_v(visual_feats)
        depth_feats = self.shallow_lnorm_d(depth_feats)
        
        # #? middle layer
        for i, l in enumerate(self.middle_att):
            visual_feats, depth_feats = l(visual_feats, depth_feats)
        
        visual_feats = self.middle_lnorm_v(visual_feats)
        
        # vanilla transformer test
        # visual_feats = torch.cat([visual_feats, depth_feats], dim=-2)
        # for l in self.parallel_att:
        #     visual_feats = l(visual_feats)
        return visual_feats, depth_feats