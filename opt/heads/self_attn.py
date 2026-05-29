# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# heads/self_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SelfAttnLayerConfig:
    """Configuration for SelfAttnLayer"""
    block_num: int = 2
    d_model: int = 256
    num_head: int = 4
    dim_ffn: int = 256


class SelfAttnBlock(nn.Module):
    """Self-attention block with feed-forward network"""
    
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
            
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(), 
            nn.Dropout(dropout, inplace=False),
            nn.Linear(dim_ffn, d_model)
        )
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    def forward(self, kpt_query):
        # self-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, _ = self.self_attn(
            kpt_query2, 
            kpt_query2, 
            value=kpt_query2
        )
        kpt_query = kpt_query + self.dropout1(kpt_query2)
        
        # ffn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout2(kpt_query2)
        
        return kpt_query


class SelfAttnLayer(nn.Module):
    """
    Self-attention layer composed of multiple self-attention blocks.
    Used for feature refinement in NOCS prediction.
    """
    
    def __init__(self, cfg: SelfAttnLayerConfig):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn
        
        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(
                SelfAttnBlock(
                    d_model=self.d_model, 
                    num_heads=self.num_head, 
                    dim_ffn=self.dim_ffn, 
                    dropout=0.0, 
                    dropout_attn=None
                )
            )
        
    def forward(self, batch_kpt_query):
        """
        Args:
            batch_kpt_query: (b, kpt_num, dim)
            
        Returns:
            batch_kpt_query: (b, kpt_num, dim)
        """    
        for i in range(self.block_num):
            batch_kpt_query = self.attn_blocks[i](batch_kpt_query)
        
        return batch_kpt_query

