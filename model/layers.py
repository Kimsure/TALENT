import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import fvcore.nn.weight_init as weight_init
# from .DWT import DWT_2D, IDWT_2D

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def deconv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.ReLU(True))

class CostVolume(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.txt_lin = linear_layer(d_model, d_model)
        self.out_conv = conv_layer(1, d_model)
        
    def forward(self, vis, text):
        query = vis                     # [B,D,H,W]
        # print("query.minmax: ", query.min(), query.max())
        text = text.permute(1, 0, 2)    # [1,B,D] -> [B,1,D]
        value = self.txt_lin(text)      # [B,1,D]
        # cost volume
        out = torch.einsum("bchw, blc -> blhw", query, value)
        # print("costVolue.shape", out.shape)
        # print("costVolue.minmax: ", out.min(), out.max())
        tgt = self.out_conv(out)
        # tgt = F.softmax(tgt, dim=1)  # Normalize along the last dimension
        tgt = tgt * query
        # print("tgt.minmax: ", tgt.min(), tgt.max())
        # print("tgt.shape", tgt.shape)
        return tgt
    
class CrossAttn(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.d_model = d_model
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        query=self.with_pos_embed(tgt, query_pos)
        key=self.with_pos_embed(memory, pos)
        value=memory
        value = value.to(dtype=torch.float32)
        # **用 MultiheadAttention 计算原始 attn_weight**
        tgt2, attn_weight = self.multihead_attn(query=query, key=key, value=value, attn_mask=None,
                                                key_padding_mask=memory_key_padding_mask,
                                                average_attn_weights=True)
        
        tgt = tgt * tgt2
        return tgt, attn_weight


class Neck(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024],
                 stride = [1, 1, 1], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(Neck, self).__init__()
        self.fusion3 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion4 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion5 = CrossAttn(d_model=d_model, nhead=nhead)     
        # self.txt_proj = nn.Linear(in_channels[2], out_channels[1])   
        self.txt_proj = nn.Linear(512, out_channels[1])
        self.f3_proj = conv_layer(in_channels[0], out_channels[1], 1, 0, stride[0])
        self.f4_proj = conv_layer(in_channels[1], out_channels[1], 1, 0, stride[1])
        self.f5_proj = deconv_layer(in_channels[2], out_channels[1], 1, 0, stride[2])
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        # self.dwt_v3 = DWT_2D(wavename='haar')
        # self.dwt_v4 = DWT_2D(wavename='haar')
        # self.dwt_v5 = DWT_2D(wavename='haar')
        # self.softmax = nn.Softmax2d()
        
    def forward(self, imgs, state):
        # v3, v4, v5: 512, 52, 52 / 1024, 26, 26 / 512, 13, 13
        # v3 = imgs[0]
        # v4 = v3.clone()
        # v5 = v3.clone()
        v3, v4, v5 = imgs
        # print("v3 v4 v5.shape", v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)
        # wavelet deband and pooling
        if 0:
            v3_ll, v3_lh, v3_hl, _ = self.dwt_v3(v3)
            v4_ll, v4_lh, v4_hl, _ = self.dwt_v3(v4)
            v5_ll, v5_lh, v5_hl, _ = self.dwt_v3(v5)
            # print("v3_ll lh hl.shape", v3_ll.shape, v3_lh.shape, v3_hl.shape)
            v3_high = self.softmax(torch.add(v3_lh, v3_hl))
            v4_high = self.softmax(torch.add(v4_lh, v4_hl))
            v5_high = self.softmax(torch.add(v5_lh, v5_hl))
            v3_attn = torch.mul(v3_ll, v3_high)
            v4_attn = torch.mul(v4_ll, v4_high)
            v5_attn = torch.mul(v5_ll, v5_high)
            v3_ll = torch.add(v3_ll, v3_attn)
            v4_ll = torch.add(v4_ll, v4_attn)
            v5_ll = torch.add(v5_ll, v5_attn)
        
        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)
        txt = self.txt_proj(txt)
        vis_out = [v3, v4, v5] # for visualization
        Neck_attn_weights = []
        # fusion v3 
        b, c, h, w = v3.shape
        v3 = v3.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c
        # fusion v4 
        b, c, h, w = v4.shape
        v4 = v4.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c  
        # fusion v5 
        b, c, h, w = v5.shape
        v5 = v5.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c       
        
        
        fq3, attn_weight_3 = self.fusion3(v3, txt)         
        fq3 = fq3.permute(1, 2, 0).reshape(b, c, h, w)
        Neck_attn_weights.append(attn_weight_3)   
        # v4 = self.downsample(v4)
        fq4, attn_weight_4 = self.fusion4(v4, txt)         
        fq4 = fq4.permute(1, 2, 0).reshape(b, c, h, w)
        Neck_attn_weights.append(attn_weight_4)
        fq5, attn_weight_5 = self.fusion5(v5, txt) 
        fq5 = fq5.permute(1, 2, 0).reshape(b, c, h, w)
        Neck_attn_weights.append(attn_weight_5)
        vis_out.append(fq3)
        vis_out.append(fq4)
        vis_out.append(fq5)
        # fusion 4: b, 512, 26, 26 / b, 512, 26, 26 / b, 512, 26, 26
        # query
        fq = torch.cat([fq3, fq4, fq5], dim=1)

        fq = self.aggr(fq)
        vis_out.append(fq)
        fq1 = self.coordconv(fq)

        fq = fq1 + fq
        vis_out.append(fq)
        if 0:
            vis_out.append(v3_ll)
            vis_out.append(v4_ll)
            vis_out.append(v5_ll)
        # b, 512, 26, 26
        return fq, Neck_attn_weights, vis_out
    
    
class Neck_CostVolume(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024],
                 stride = [1, 1, 1], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(Neck_CostVolume, self).__init__()
        self.fusion3 = CostVolume(d_model=d_model, nhead=nhead)
        self.fusion4 = CostVolume(d_model=d_model, nhead=nhead)
        self.fusion5 = CostVolume(d_model=d_model, nhead=nhead)     
        # self.txt_proj = nn.Linear(in_channels[2], out_channels[1])   
        self.txt_proj = nn.Linear(512, out_channels[1])
        self.f3_proj = conv_layer(in_channels[0], out_channels[1], 1, 0, stride[0])
        self.f4_proj = conv_layer(in_channels[1], out_channels[1], 1, 0, stride[1])
        self.f5_proj = deconv_layer(in_channels[2], out_channels[1], 1, 0, stride[2])
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        
    def forward(self, imgs, state):
        v3, v4, v5 = imgs
        # print("v3 v4 v5.shape", v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)        
        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)
        txt = self.txt_proj(txt)
        # print("Neck_cost_txt.shape", txt.shape) # [L,B,D]
        vis_out = [v3, v4, v5] # for visualization
        Neck_attn_weights = []     
          
        fq3 = self.fusion3(v3, txt)         
        fq4 = self.fusion4(v4, txt)         
        fq5 = self.fusion5(v5, txt) 
        vis_out.append(fq3)
        vis_out.append(fq4)
        vis_out.append(fq5)
        # fusion 4: b, 512, 26, 26 / b, 512, 26, 26 / b, 512, 26, 26
        # query
        fq = torch.cat([fq3, fq4, fq5], dim=1)

        fq = self.aggr(fq)
        vis_out.append(fq)
        fq1 = self.coordconv(fq)

        fq = fq1 + fq
        vis_out.append(fq)
        # b, 512, 26, 26
        return fq, Neck_attn_weights, vis_out
    

class Neck_CostAffinity(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024],
                 stride = [1, 1, 1], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(Neck_CostAffinity, self).__init__()
        self.fusion3 = CostVolume(d_model=d_model, nhead=nhead)
        self.fusion4 = CostVolume(d_model=d_model, nhead=nhead)
        self.fusion5 = CostVolume(d_model=d_model, nhead=nhead)      
        self.txt_proj = nn.Linear(512, out_channels[1])
        self.f3_proj = conv_layer(in_channels[0], out_channels[1], 1, 0, stride[0])
        self.f4_proj = conv_layer(in_channels[1], out_channels[1], 1, 0, stride[1])
        self.f5_proj = deconv_layer(in_channels[2], out_channels[1], 1, 0, stride[2])
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        
    def forward(self, imgs, state):
        v3, v4, v5 = imgs
        # print("v3 v4 v5.shape", v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)        
        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)
        txt = self.txt_proj(txt)
        # print("Neck_cost_txt.shape", txt.shape) # [L,B,D]
        vis_out = [v3, v4, v5] # for visualization
        Neck_attn_weights = []     
        if 1:
            fq3 = self.fusion3(v3, txt)         
            fq4 = self.fusion4(v4, txt)         
            fq5 = self.fusion5(v5, txt) 
        else:
            fq3 = v3
            fq4 = v4
            fq5 = v5
        vis_out.append(fq3)
        vis_out.append(fq4)
        vis_out.append(fq5)
        # print("fq3 fq4 fq5.shape", fq3.shape, fq4.shape, fq5.shape)
        if self.training:
            # for affinity loss calculation
            # Calculate cosine similarity on fq3
            b, c, h, w = fq3.shape
            fq3_flat = fq3.view(b, c, -1)  # b, c, hw
            fq4_flat = fq4.view(b, c, -1)  # b, c, hw
            fq5_flat = fq5.view(b, c, -1)  # b, c, hw
            fq3_norm = F.normalize(fq3_flat, p=2, dim=1)  # normalize along channel dimension
            fq4_norm = F.normalize(fq4_flat, p=2, dim=1)  # normalize along channel dimension
            fq5_norm = F.normalize(fq5_flat, p=2, dim=1)  # normalize along channel dimension
            fq3_affinity = torch.einsum('bch,bcH->bhH', fq3_norm, fq3_norm)  # b, hw, hw
            fq4_affinity = torch.einsum('bch,bcH->bhH', fq4_norm, fq4_norm)  # b, hw, hw
            fq5_affinity = torch.einsum('bch,bcH->bhH', fq5_norm, fq5_norm)  # b, hw, hw
            fq3_affinity = F.softmax(fq3_affinity, dim=-1)  # Normalize along the last dimension
            fq4_affinity = F.softmax(fq4_affinity, dim=-1)  # Normalize along the last dimension 
            fq5_affinity = F.softmax(fq5_affinity, dim=-1)  # Normalize along the last dimension
            # Calculate cosine similarity between text and feature vectors
            txt = txt.permute(1, 2, 0)  # [L,B,D] -> [B,D,L]
            B, D, L = txt.shape
            txt_norm = F.normalize(txt, p=2, dim=1)  # Normalize along channel dimension
            # Calculate cosine similarity with each feature level using einsum
            fq3_txt_sim = torch.einsum('bcs,bcl->bsl', fq3_flat, txt)  # B,HW,L
            fq4_txt_sim = torch.einsum('bcs,bcl->bsl', fq4_flat, txt)  # B,HW,L
            fq5_txt_sim = torch.einsum('bcs,bcl->bsl', fq5_flat, txt)  # B,HW,L
            
            fq3_lvaffinity = torch.einsum('bsl,bSl->bsS', fq3_txt_sim, fq3_txt_sim)  # B,HW,HW
            fq4_lvaffinity = torch.einsum('bsl,bSl->bsS', fq4_txt_sim, fq4_txt_sim)  # B,HW,HW
            fq5_lvaffinity = torch.einsum('bsl,bSl->bsS', fq5_txt_sim, fq5_txt_sim)  # B,HW,HW
            # print("fq3_lvaffinity shapes:", fq3_lvaffinity.shape)
            fq3_lvaffinity = F.softmax(fq3_lvaffinity, dim=-1)  # Normalize along the last dimension
            fq4_lvaffinity = F.softmax(fq4_lvaffinity, dim=-1)
            fq5_lvaffinity = F.softmax(fq5_lvaffinity, dim=-1)
            # print("fq_flat minmax:", fq3_flat.min(), fq3_flat.max(), 
            #       fq4_flat.min(), fq4_flat.max(), 
            #       fq5_flat.min(), fq5_flat.max())
            # print("txt minmiax", txt.min(), txt.max())
            # print("fq_lvaffinity minmax:", fq3_lvaffinity.min(), fq3_lvaffinity.max(), 
            #       fq4_lvaffinity.min(), fq4_lvaffinity.max(), 
            #       fq5_lvaffinity.min(), fq5_lvaffinity.max())
            fq3_lvaffinity = (fq3_lvaffinity > 0.7).float()  # Thresholding to create binary affinity
            fq4_lvaffinity = (fq4_lvaffinity > 0.7).float()  # Thresholding to create binary affinity
            fq5_lvaffinity = (fq5_lvaffinity > 0.7).float()  # Thresholding to create binary affinity
            # print("fq5_lvaffinity shapes:", fq5_lvaffinity.shape, fq5_lvaffinity.min(), fq5_lvaffinity.max())
            # Element-wise multiplication between fq3_affinity and fq3_lvaffinity
            fq3_affinity = torch.mul(fq3_affinity, fq3_lvaffinity)
            fq4_affinity = torch.mul(fq4_affinity, fq4_lvaffinity)
            fq5_affinity = torch.mul(fq5_affinity, fq5_lvaffinity)
            # Calculate L2/Frobenius norm loss between fq3_affinity and fq3_lvaffinity
            fq3_affinity_loss=(fq3_affinity - fq3_lvaffinity).norm(p=2,dim=-1).mean() 
            fq4_affinity_loss=(fq4_affinity - fq4_lvaffinity).norm(p=2,dim=-1).mean()
            fq5_affinity_loss=(fq5_affinity - fq5_lvaffinity).norm(p=2,dim=-1).mean()
            # Average the losses
            affinity_loss = (fq3_affinity_loss + fq4_affinity_loss + fq5_affinity_loss) / 3.0
            
        
        # fusion 4: b, 512, 26, 26 / b, 512, 26, 26 / b, 512, 26, 26
        # query
        fq = torch.cat([fq3, fq4, fq5], dim=1)

        fq = self.aggr(fq)
        vis_out.append(fq)
        fq1 = self.coordconv(fq)

        fq = fq1 + fq
        vis_out.append(fq)
        # b, 512, 26, 26
        if self.training:
            return fq, Neck_attn_weights, vis_out, affinity_loss
        else:
            return fq, Neck_attn_weights, vis_out
        
        
class Neck_CrossAffinity(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024],
                 stride = [1, 1, 1], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(Neck_CrossAffinity, self).__init__()
        self.fusion3 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion4 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion5 = CrossAttn(d_model=d_model, nhead=nhead)      
        self.txt_proj = nn.Linear(512, out_channels[1])
        self.f3_proj = conv_layer(in_channels[0], out_channels[1], 1, 0, stride[0])
        self.f4_proj = conv_layer(in_channels[1], out_channels[1], 1, 0, stride[1])
        self.f5_proj = deconv_layer(in_channels[2], out_channels[1], 1, 0, stride[2])
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        
    def forward(self, imgs, state):
        v3, v4, v5 = imgs
        # print("v3 v4 v5.shape", v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)        
        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)
        txt = self.txt_proj(txt)
        # print("Neck_cost_txt.shape", txt.shape) # [L,B,D]
        vis_out = [v3, v4, v5] # for visualization
        Neck_attn_weights = []     
        if 1:
            fq3 = self.fusion3(v3, txt)         
            fq4 = self.fusion4(v4, txt)         
            fq5 = self.fusion5(v5, txt) 
        else:
            fq3 = v3
            fq4 = v4
            fq5 = v5
        vis_out.append(fq3)
        vis_out.append(fq4)
        vis_out.append(fq5)
        # print("fq3 fq4 fq5.shape", fq3.shape, fq4.shape, fq5.shape)
        if self.training:
            # for affinity loss calculation
            # Calculate cosine similarity on fq3
            b, c, h, w = fq3.shape
            fq3_flat = fq3.view(b, c, -1)  # b, c, hw
            fq4_flat = fq4.view(b, c, -1)  # b, c, hw
            fq5_flat = fq5.view(b, c, -1)  # b, c, hw
            fq3_norm = F.normalize(fq3_flat, p=2, dim=1)  # normalize along channel dimension
            fq4_norm = F.normalize(fq4_flat, p=2, dim=1)  # normalize along channel dimension
            fq5_norm = F.normalize(fq5_flat, p=2, dim=1)  # normalize along channel dimension
            fq3_affinity = torch.einsum('bch,bcH->bhH', fq3_norm, fq3_norm)  # b, hw, hw
            fq4_affinity = torch.einsum('bch,bcH->bhH', fq4_norm, fq4_norm)  # b, hw, hw
            fq5_affinity = torch.einsum('bch,bcH->bhH', fq5_norm, fq5_norm)  # b, hw, hw
            fq3_affinity = F.softmax(fq3_affinity, dim=-1)  # Normalize along the last dimension
            fq4_affinity = F.softmax(fq4_affinity, dim=-1)  # Normalize along the last dimension 
            fq5_affinity = F.softmax(fq5_affinity, dim=-1)  # Normalize along the last dimension
            # Calculate cosine similarity between text and feature vectors
            txt = txt.permute(1, 2, 0)  # [L,B,D] -> [B,D,L]
            B, D, L = txt.shape
            txt_norm = F.normalize(txt, p=2, dim=1)  # Normalize along channel dimension
            # Calculate cosine similarity with each feature level using einsum
            fq3_txt_sim = torch.einsum('bcs,bcl->bsl', fq3_flat, txt)  # B,HW,L
            fq4_txt_sim = torch.einsum('bcs,bcl->bsl', fq4_flat, txt)  # B,HW,L
            fq5_txt_sim = torch.einsum('bcs,bcl->bsl', fq5_flat, txt)  # B,HW,L
            
            fq3_lvaffinity = torch.einsum('bsl,bSl->bsS', fq3_txt_sim, fq3_txt_sim)  # B,HW,HW
            fq4_lvaffinity = torch.einsum('bsl,bSl->bsS', fq4_txt_sim, fq4_txt_sim)  # B,HW,HW
            fq5_lvaffinity = torch.einsum('bsl,bSl->bsS', fq5_txt_sim, fq5_txt_sim)  # B,HW,HW
            # print("fq3_lvaffinity shapes:", fq3_lvaffinity.shape)
            fq3_lvaffinity = F.softmax(fq3_lvaffinity, dim=-1)  # Normalize along the last dimension
            fq4_lvaffinity = F.softmax(fq4_lvaffinity, dim=-1)
            fq5_lvaffinity = F.softmax(fq5_lvaffinity, dim=-1)
            # print("fq_flat minmax:", fq3_flat.min(), fq3_flat.max(), 
            #       fq4_flat.min(), fq4_flat.max(), 
            #       fq5_flat.min(), fq5_flat.max())
            # print("txt minmiax", txt.min(), txt.max())
            # print("fq_lvaffinity minmax:", fq3_lvaffinity.min(), fq3_lvaffinity.max(), 
            #       fq4_lvaffinity.min(), fq4_lvaffinity.max(), 
            #       fq5_lvaffinity.min(), fq5_lvaffinity.max())
            fq3_lvaffinity = (fq3_lvaffinity > 0.7).float()  # Thresholding to create binary affinity
            fq4_lvaffinity = (fq4_lvaffinity > 0.7).float()  # Thresholding to create binary affinity
            fq5_lvaffinity = (fq5_lvaffinity > 0.7).float()  # Thresholding to create binary affinity
            # print("fq5_lvaffinity shapes:", fq5_lvaffinity.shape, fq5_lvaffinity.min(), fq5_lvaffinity.max())
            # Element-wise multiplication between fq3_affinity and fq3_lvaffinity
            fq3_affinity = torch.mul(fq3_affinity, fq3_lvaffinity)
            fq4_affinity = torch.mul(fq4_affinity, fq4_lvaffinity)
            fq5_affinity = torch.mul(fq5_affinity, fq5_lvaffinity)
            # Calculate L2/Frobenius norm loss between fq3_affinity and fq3_lvaffinity
            fq3_affinity_loss=(fq3_affinity - fq3_lvaffinity).norm(p=2,dim=-1).mean() 
            fq4_affinity_loss=(fq4_affinity - fq4_lvaffinity).norm(p=2,dim=-1).mean()
            fq5_affinity_loss=(fq5_affinity - fq5_lvaffinity).norm(p=2,dim=-1).mean()
            # Average the losses
            affinity_loss = (fq3_affinity_loss + fq4_affinity_loss + fq5_affinity_loss) / 3.0
            
        
        # fusion 4: b, 512, 26, 26 / b, 512, 26, 26 / b, 512, 26, 26
        # query
        fq = torch.cat([fq3, fq4, fq5], dim=1)

        fq = self.aggr(fq)
        vis_out.append(fq)
        fq1 = self.coordconv(fq)

        fq = fq1 + fq
        vis_out.append(fq)
        # b, 512, 26, 26
        if self.training:
            return fq, Neck_attn_weights, vis_out, affinity_loss
        else:
            return fq, Neck_attn_weights, vis_out
        
 
class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class Decoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=None)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



