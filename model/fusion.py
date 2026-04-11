import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
import numpy as np
import torch.distributed as dist
from .layers import conv_layer, deconv_layer
import os
from functools import partial, reduce
from operator import mul
from torch.nn import LayerNorm, Dropout 


class Fusion(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 dino_layers= 12,
                 output_dinov2 =[4, 8] ,
                ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0
        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        if 0:
            self.interact_t2v = nn.Sequential(
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Linear(32, 768)
            )
            # self.interact_v2t = nn.Sequential(
            #     nn.Linear(768, 32),
            #     nn.ReLU(),
            #     nn.Linear(32, 512)
            # )
        if 0:    
            self.num_tokens = 16
            self.total_d_layer = 0
            self._init_prompt(num_tokens = self.num_tokens, prompt_dim = self.d_txt, total_d_layer = self.total_d_layer)
                
        self.initialize_parameters()

    def _init_prompt(self, num_tokens, prompt_dim, total_d_layer):
        # 计算 xavier_uniform 初始化范围
        val = math.sqrt(6. / float(num_tokens + prompt_dim))

        if total_d_layer >= 0:
            # 浅层 prompt（插入到输入序列中）
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:
                # 深层 prompt：比如每层 transformer 一个可学习的 prompt（类比 DeepPrompt）
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, num_tokens, prompt_dim)
                )
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            # projection 层和归一化
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
            self.prompt_dropout = Dropout(0.1)
            print("initialize prompt drop")

        else:
            # total_d_layer < 0 的情况（可理解为纯 deep prompt 模式）
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(abs(total_d_layer), num_tokens, prompt_dim)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
            self.prompt_dropout = Dropout(0.1)
            print("initialize prompt drop")
            
    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, img, text, txt_backbone,dino,pad_mask=None):
        B=img.shape[0]
        img = img.type(txt_backbone.dtype)
        vis_outs = []
        outputs=[]
        txt = txt_backbone.token_embedding(text).type(
            txt_backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = txt_backbone.transformer
        txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]
        
        #dinov2  
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat((dino.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
        dino_f = dino_f + dino.interpolate_pos_encoding(dino_f, w, h)
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(dino_f.shape[0], -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )
        
        # language
        if 0:
            if self.interact_v2t:
                txt_add = self.interact_v2t(dino_f[:, 0:1, :])
                txt = torch.cat((txt_add, txt), dim=1)
            
        # text prompt tunning
        if 0:
            B, L, D = txt.shape
            txt = torch.cat((
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    txt[:, :, :]
                ), dim=1)
                
        txt = txt.permute(1, 0, 2)  # BLD -> LBD   
        # text prompt tunning
        if 0:
            if self.total_d_layer == 0: #shallow 
                for i in range(self.num_layers):
                    txt = txt_enc.resblocks[i](txt)
                txt = txt[self.num_tokens:, :, :]
            elif self.total_d_layer > 0: # deep
                for i in range(self.num_layers):
                    if i == 0:
                        txt = txt_enc.resblocks[i](txt)
                    elif i <= self.deep_prompt_embeddings.shape[0]:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                        txt = torch.cat((
                            deep_prompt_emb,
                            txt[self.num_tokens:, :, :]
                        ), dim=0)
                        txt = txt_enc.resblocks[i](txt)
                    else:
                        txt = txt[-L:, :, :]
                        txt = txt_enc.resblocks[i](txt)
                txt = txt[-L:, :, :]
            else:
                AttributeError('Input correct total_d_layer')
        # no text prompt tunning
        else:
            for i in range(self.num_layers):
                txt = txt_enc.resblocks[i](txt)
                
        txt = txt.permute(1, 0, 2)  # LBD -> BLD
        txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # txt = txt[:, :-1, :]
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ txt_backbone.text_projection# get sentence-level feature Fs
        
        
        # visual
        features_dino=[]
        if 0:
            if self.interact_t2v:
                dino_f = torch.cat(
                    (
                        dino_f[:, 0:5, :], 
                        self.interact_t2v(txt),
                        dino_f[:, 5:, :]
                    ), 
                    dim=1,
                )
        vis_feats = []
        for i in range(self.dino_layers):
            dino_f = dino.blocks[i](dino_f, txt, pad_mask)
            vis_feats.append(dino_f)
            if i in self.output_dinov2:
                features_dino.append(dino_f)
        
        dino_f = dino.norm(dino_f)
        features_dino.append(dino_f)
        

        for i, feature_dino in enumerate(features_dino):
            # print("feature_dino.shape:", feature_dino.shape)
            feature_dino=feature_dino[:, 4 + 1:]
            B,L,C = feature_dino.shape
            H = int(L ** 0.5)
            W = L // H
            feature_dino = feature_dino.reshape(B, H, W, C).permute(0, 3, 1, 2)

            vis_outs.append(feature_dino)
            
        for i, feature in enumerate(vis_feats):
            feature=feature[:, 4 + 1:]
            B,L,C = feature.shape
            H = int(L ** 0.5)
            W = L // H
            feature = feature.reshape(B, H, W, C).permute(0, 3, 1, 2)
            vis_feats[i] = feature
 

        # forward

        output = vis_outs , txt, state, vis_feats

        return output





class Fusion_EVA(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 dino_layers= 12,
                 output_dinov2 =[4, 8] ,
                ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        if 1:
            self.interact_t2v = nn.Sequential(
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Linear(32, 768)
            )
            # self.interact_v2t = nn.Sequential(
            #     nn.Linear(768, 32),
            #     nn.ReLU(),
            #     nn.Linear(32, 512)
            # )
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, img, text, txt_backbone,dino):
        B=img.shape[0]
        img = img.type(txt_backbone.dtype)
        vis_outs = []
        outputs=[]
        txt_enc = txt_backbone.text
        cast_dtype = txt_enc.transformer.get_cast_dtype()
        txt = txt_enc.token_embedding(text).to(cast_dtype)   # [batch_size, n_ctx, d_model]
        txt = txt + txt_enc.positional_embedding.to(cast_dtype)[:txt.size(1)]
        
        #dinov2  
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat((dino.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
        dino_f = dino_f + dino.interpolate_pos_encoding(dino_f, w, h)
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(dino_f.shape[0], -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )
        if 1:
            # if self.interact_v2t:
            #     txt_add = self.interact_v2t(dino_f[:, 0:1, :])
            #     txt = torch.cat((txt_add, txt), dim=1)
            if self.interact_t2v:
                dino_f = torch.cat(
                    (
                        dino_f[:, 0:5, :], 
                        self.interact_t2v(txt),
                        dino_f[:, 5:, :]
                    ), 
                    dim=1,
                )
            
        txt = txt.permute(1, 0, 2)  # BLD -> LBD    
        features_dino=[]
        for i in range(self.num_layers):
            txt = txt_enc.transformer.resblocks[i](txt)

        # language
        txt = txt.permute(1, 0, 2)  # LBD -> BLD
        txt = txt_enc.ln_final(txt).type(cast_dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ txt_enc.text_projection# get sentence-level feature Fs
        
        for i in range(self.dino_layers):
            dino_f = dino.blocks[i](dino_f, txt)
            if i in self.output_dinov2:
                features_dino.append(dino_f)
        
        dino_f = dino.norm(dino_f)
        features_dino.append(dino_f)
        

        for i, feature_dino in enumerate(features_dino):
            feature_dino=feature_dino[:, 4 + 1 + 17:]
            B,L,C = feature_dino.shape
            H = int(L ** 0.5)
            W = L // H
            feature_dino = feature_dino.reshape(B, H, W, C).permute(0, 3, 1, 2)

            vis_outs.append(feature_dino)
 

        # forward

        output = vis_outs , txt, state

        return output
