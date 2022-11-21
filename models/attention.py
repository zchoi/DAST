'''
    pytorch implementation of our RMN model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gumbel as gumbel
# from models.allennlp_beamsearch import BeamSearch
import math

from configparser import InterpolationMissingOptionError
from curses import intrflush
from pickle import NONE
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from contextlib import contextmanager
from typing import Union, Sequence, Tuple

TensorOrNone = Union[torch.Tensor, None]
# ------------------------------------------------------
# ------------ Soft Attention Mechanism ----------------
# ------------------------------------------------------

class SoftAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size, dropout=0.1):
        super(SoftAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

    def forward(self, feats, f=None, key=None):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha

# ------------------------------------------------------
# ------------ Self Attention Mechanism --------------
# ------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.q_liner = nn.Linear(feat_size, hidden_size)
        self.k_liner = nn.Linear(feat_size, hidden_size)
        self.v_liner = nn.Linear(feat_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None, dropout=None):
        q, k, v = self.q_liner(input), self.k_liner(input), self.v_liner(input)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn

# ------------------------------------------------------
# ------------ MY Attention Mechanism --------------
# ------------------------------------------------------
class MYAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size, dropout=0.1):
        super(MYAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

        self.wh2 = nn.Linear(hidden_size, att_size)
        self.wv2 = nn.Linear(feat_size, att_size)
        self.wa2 = nn.Linear(att_size, 1, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(att_size*2 , att_size,),
            nn.Dropout(p=.1),
        )
        
        self.fc_sigmoid = nn.Sequential(
            nn.Linear(feat_size*3, feat_size),
            nn.Sigmoid()
        )

    def forward(self, feats, cluster_feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param cluster_feats: (batch_size, center_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        
        v2 = self.wv2(cluster_feats)
        inputs2 = self.wh2(att_feats).unsqueeze(1).expand_as(v2) + v2
        alpha2 = F.softmax(self.wa2(torch.tanh(inputs2)).squeeze(-1), dim=1)
        att_feats2 = torch.bmm(alpha2.unsqueeze(1), cluster_feats).squeeze(1)

        # gate_info = torch.cat([att_feats, key, att_feats2], dim=-1)
        # gate_info = self.fc_sigmoid(gate_info)
        
        # return gate_info * att_feats + (1 - gate_info) * att_feats2, None
        return att_feats + att_feats2, None


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._is_stateful = False
        self._state_names = []
        self._state_defaults = dict()

    def register_state(self, name: str, default: TensorOrNone):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = default.clone().detach()
        self.register_buffer(name, default)

    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.children():
            if isinstance(m, Module):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])
        for m in self.children():
            if isinstance(m, Module):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)
                self._buffers[name] = self._buffers[name].unsqueeze(0)
                self._buffers[name] = self._buffers[name].expand([batch_size, ] + list(self._buffers[name].shape[1:]))
                self._buffers[name] = self._buffers[name].contiguous()

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)

    def enable_statefulness(self, batch_size: int):
        for m in self.children():
            if isinstance(m, Module):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        for m in self.children():
            if isinstance(m, Module):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False

    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.video_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.depth_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, video_feats=None, depth_feats=None, mask_queries = None):
        
        # MHA+AddNorm
        # test
        self_att = self.self_att(input, input, input, None)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        # # MHA+AddNorm:Image
        # v_att = self.video_att(self_att, video_feats, video_feats, None)
        # d_att = self.depth_att(self_att, depth_feats, depth_feats, None)

        # enc_att = self.lnorm2(self_att + self.dropout2(v_att) + self.dropout3(d_att))
        enc_att = self.lnorm2(self_att)
        if mask_queries is not None:
            enc_att = enc_att * mask_queries
        ff = self.pwff(enc_att)

        return ff


class ShallowLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(ShallowLayer, self).__init__()
        self.shared_MHA = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.separated_MHA_video = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.separated_MHA_depth = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.shared_lnorm = nn.LayerNorm(d_model)
        self.lnorm1_d = nn.LayerNorm(d_model)
        self.lnorm2_v = nn.LayerNorm(d_model)
        self.lnorm2_d = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, video_feats, depth_feats, mask_queries = None):
        
        # MHA+AddNorm
        # video_att = self.shared_MHA(video_feats, video_feats, video_feats, None)
        # depth_att = self.shared_MHA(depth_feats, depth_feats, depth_feats, None)
        
        # video_ = self.shared_lnorm(video_feats + self.dropout1(video_att))
        # depth_ = self.shared_lnorm(depth_feats + self.dropout2(depth_att))

        # # MHA+AddNorm:Image
        video_ = video_feats
        depth_ = depth_feats
        # video_out = self.separated_MHA_video(video_, video_, video_, None)
        # depth_out = self.separated_MHA_depth(depth_, depth_, depth_, None)

        # video_out = self.lnorm2_v(video_ + self.dropout3(video_out))
        # depth_out = self.lnorm2_d(depth_ + self.dropout4(depth_out))
        
        vff = self.pwff(video_)
        dff = self.pwff(depth_)
        
        return vff, dff

class depth_guided_block(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(depth_guided_block, self).__init__()

        self.MHA = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.d_model = d_model
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, video_feat, depth_feat, mask_queries=None, permute=True):
        
        if permute:
            video_feat, depth_feat = video_feat.permute(0, 2, 1), depth_feat.permute(0, 2, 1) 
        
        bs, num_f, dim = video_feat.size()
        _, num_d, _ = depth_feat.size()
        
        # simple implementation for test
        enc_mask = torch.ones(num_f, 1, 1, num_d).bool().to(depth_feat.device)
        neighbor_frame = 8
        for i in range(num_f):
            if i == 0:
                enc_mask[i, :, :, i] = 0
                enc_mask[i, :, :, i+1 : i+neighbor_frame+1] = 0
            elif i == num_d-1:
                enc_mask[i, :, :, i] = 0
                enc_mask[i, :, :, i-neighbor_frame : i] = 0
            else:
                enc_mask[i, :, :, i] = 0
                enc_mask[i, :, :, i-neighbor_frame+1 : i] = 0
                enc_mask[i, :, :, i+1 : i+neighbor_frame] = 0

        enc_mask = enc_mask.unsqueeze(0).repeat(bs, 1, 1, 1, 1).reshape(bs*num_f, 1, 1, num_d)
        
        # MHA+AddNorm:Image
        v_att = self.MHA(video_feat, depth_feat, depth_feat, enc_mask) # enc_mask

        enc_att = self.lnorm2(video_feat + self.dropout2(v_att))
        
        if mask_queries is not None:
            enc_att = enc_att * mask_queries
        # enc_att = enc_att.reshape(bs, num_f, dim)

        ff = self.pwff(enc_att)
        
        return ff.permute(0, 2, 1), depth_feat.permute(0, 2, 1)


class middle_layer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(middle_layer, self).__init__()

        self.patch_merge = nn.Conv2d(26, 26, kernel_size=1, stride=2)
        
        self.channel_merging1 = nn.Conv1d(in_channels=d_model, out_channels=d_model//2, kernel_size=3, stride=2)
        self.channel_merging2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//4, kernel_size=3, stride=2)

        self.channel_expanding1 = nn.ConvTranspose1d(in_channels=d_model//4, out_channels=d_model//2, kernel_size=3, stride=2)
        self.channel_expanding2 = nn.ConvTranspose1d(in_channels=d_model//2, out_channels=d_model, kernel_size=3, stride=2)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.depth_guided_block_512 = depth_guided_block(d_model=d_model, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1)
        self.depth_guided_block_512_2 = depth_guided_block(d_model=d_model, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1)
        
        self.depth_guided_block_256 = depth_guided_block(d_model=d_model//2, d_k=32, d_v=32, h=4, d_ff=1024, dropout=.1)
        self.depth_guided_block_256_2 = depth_guided_block(d_model=d_model//2, d_k=32, d_v=32, h=4, d_ff=1024, dropout=.1)
        
        self.depth_guided_block_128_1 = depth_guided_block(d_model=d_model//4, d_k=16, d_v=16, h=2, d_ff=512, dropout=.1)
        self.depth_guided_block_128_2 = depth_guided_block(d_model=d_model//4, d_k=16, d_v=16, h=2, d_ff=512, dropout=.1)
        # self.depth_guided_block_128_3 = depth_guided_block(d_model=128, d_k=16, d_v=16, h=2, d_ff=512, dropout=.1)

    def forward(self, video_feat, depth_feat):
        
        """
        Args:
            video_feat: [Tensor] (batch_size, num_f, module_dim)
            depth_feat: [Tensor] (batch_size, num_d, module_dim)
        Return:
            video_feat: [Tensor] (batch_size, num_f, module_dim)
            depth_feat: [Tensor] (batch_size, num_d, module_dim)
        """
        
        # block        
        video_1, depth_1 = self.depth_guided_block_512(video_feat, depth_feat, permute=False)
        video_1, depth_1 = self.depth_guided_block_512_2(video_1, depth_1, permute=True)
   
        # channel merge
        video_merged, depth_merged = self.channel_merging1(video_1), self.channel_merging1(depth_1)
        
        # block
        video_2, depth_2 = self.depth_guided_block_256(video_merged, depth_merged)
        video_2, depth_2 = self.depth_guided_block_256_2(video_2, depth_2)

        # channel merge
        video_merged, depth_merged = self.channel_merging2(video_2), self.channel_merging2(depth_2)
        
        # bottleneck
        video_3, depth_3 = self.depth_guided_block_128_1(video_merged, depth_merged)
        video_3, depth_3 = self.depth_guided_block_128_2(video_3, depth_3)
        # video_3, depth_3 = self.depth_guided_block_128_3(video_3, depth_3)

        # channel expand
        video_expanded, depth_expanded = self.channel_expanding1(video_3), self.channel_expanding1(depth_3)

        # block
        video_4, depth_4 = self.depth_guided_block_256(video_expanded + video_2, depth_expanded)
        video_4, depth_4 = self.depth_guided_block_256_2(video_4 + video_2, depth_4)
        # video_4, depth_4 = self.depth_guided_block_256(video_expanded, depth_expanded)
        # video_4, depth_4 = self.depth_guided_block_256_2(video_4, depth_4)
        
        # channel expand
        video_expanded, depth_expanded = self.channel_expanding2(video_4), self.channel_expanding2(depth_4)

        # block
        video5, depth5 = self.depth_guided_block_512(video_expanded + video_1, depth_expanded)
        video5, depth5 = self.depth_guided_block_512_2(video5 + video_1, depth5)
        # video5, depth5 = self.depth_guided_block_512(video_expanded, depth_expanded)
        # video5, depth5 = self.depth_guided_block_512_2(video5, depth5)
        
        return video5.permute(0, 2, 1) + video_feat, depth5.permute(0, 2, 1)

if __name__ == '__main__':
    pass










