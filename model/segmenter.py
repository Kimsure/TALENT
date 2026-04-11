import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.clip import build_model
from .layers import Decoder, Projector, Neck_CostAffinity
from .fusion import Fusion, Fusion_EVA
from .dinov2.models.vision_transformer import vit_base,vit_large
from utils.loss import Discriminator_loss
import eva_clip

class TALENT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Text Encoder
        if "EVA" not in cfg.clip_pretrain:
            print("using clip not EVA") 
            clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
            self.txt_backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size, cfg.txtual_adapter_layer,cfg.txt_adapter_dim).float()
            self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
        else:
            self.txt_backbone, _, _ = eva_clip.create_model_and_transforms(cfg.model_name, force_custom_clip=True, rope_size=cfg.input_size, pretrained_hf=True)
            self.fusion = Fusion_EVA(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
    
       # Fix Backbone
        for param_name, param in self.txt_backbone.named_parameters():
            if 'adapter' not in param_name : 
                param.requires_grad = False       
   

        state_dict = torch.load(cfg.dino_pretrain) 
        if cfg.dino_name=='dino-base':
            self.dinov2 = vit_base(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        else:
            self.dinov2=vit_large(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        self.dinov2.load_state_dict(state_dict, strict=False)

        for param_name, param in self.dinov2.named_parameters():
            if 'adapter' not in param_name:
                param.requires_grad = False
        
        # Multi-Modal Decoder
        self.neck = Neck_CostAffinity(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)

        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None, epoch=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        # print("img.shape", img.shape)
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis, word, state, _ = self.fusion(img, word, self.txt_backbone, self.dinov2, None)

        # b, 512, 26, 26 (C4)
        if self.training:
            fq, _, _, affinity_loss = self.neck(vis, state)
        else:
            fq, _, _ = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)
                
        # b, 1, 104, 104
        pred = self.proj(fq, state)

        if self.training:

            if pred.shape[-2:] != mask.shape[-2:]:
                pred = F.interpolate(pred, mask.shape[-2:],
                                     mode='nearest') 
                mask = mask.detach()
                
            dis_l = Discriminator_loss()
            loss_discriminator = dis_l(pred, mask)

            loss = loss_discriminator + 0.1 * affinity_loss
                
            return pred.detach(), mask, loss
        else:
            return pred.detach()
        
        
    def forward_with_features(self, img, word):
        """前向传播并返回中间特征
        Args:
            img: 输入图像 [b, 3, h, w]
            word: 输入文本 [b, words]
        Returns:
            pred: 预测结果
            features: 包含中间特征的字典
        """
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        print("word.shape", word.shape)
        # 获取视觉和文本特征
        vis, word, state, vis_feats = self.fusion(img, word, self.txt_backbone, self.dinov2, None)

        # 获取neck特征 [b, 512, 26, 26]
        Neck_fq, _, Neck_vis = self.neck(vis, state)
        b, c, h, w = Neck_fq.size()

        # 获取decoder特征
        decoder_fq = self.decoder(Neck_fq, word, pad_mask)
        decoder_fq = decoder_fq.reshape(b, c, h, w)

        # 获取最终预测结果
        pred = self.proj(decoder_fq, state)

        # 收集所有特征
        features = {
            'vis': vis,  # 视觉特征
            'vis_feats': vis_feats,  # 视觉特征列表
            'Neck_fq': Neck_fq,  # neck输出特征
            'Neck_vis': Neck_vis,  # neck输出特征
            'decoder_fq': decoder_fq,  # decoder输出特征
            'pred': pred,  # 最终预测结果
        }

        return pred.detach(), features, _, word, state