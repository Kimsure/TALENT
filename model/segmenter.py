import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.clip import build_model
from .layers import Decoder, Projector, Neck_CostAffinity
from .fusion import Fusion, Fusion_EVA
from .dinov2.models.vision_transformer import vit_base,vit_large
from utils.loss import Discriminator_loss, TCCL_loss
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
        self.discriminator_loss = Discriminator_loss()
        self.tccl_loss = TCCL_loss()

    def forward(self, img, word, mask=None, epoch=None, pos_word=None, neg_word=None, tccl_valid=None):
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
        pos_state = None
        neg_state = None
        if self.training and pos_word is not None and neg_word is not None:
            vis, word, state, pos_state, neg_state, _ = self.fusion(img,
                                                                    word,
                                                                    self.txt_backbone,
                                                                    self.dinov2,
                                                                    None,
                                                                    pos_word,
                                                                    neg_word)
        else:
            vis, word, state, _ = self.fusion(img, word, self.txt_backbone, self.dinov2, None)

        # b, 512, 26, 26 (C4)
        if self.training:
            fq, _, _, affinity_loss, tccl_feature = self.neck(vis, state)
        else:
            fq, _, _, _ = self.neck(vis, state)
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
                
            loss_discriminator = self.discriminator_loss(pred, mask)

            loss = loss_discriminator + 0.1 * affinity_loss
            if pos_state is not None and neg_state is not None:
                visual_proto = F.adaptive_avg_pool2d(tccl_feature, (1, 1)).flatten(1)
                loss_tccl = self.tccl_loss(visual_proto, pos_state, neg_state, tccl_valid)
                loss = loss + 0.1 * loss_tccl
                
            return pred.detach(), mask, loss
        else:
            return pred.detach()
        
        
    def forward_with_features(self, img, word):
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        # print("word.shape", word.shape)
        vis, word, state, vis_feats = self.fusion(img, word, self.txt_backbone, self.dinov2, None)
        neck_outputs = self.neck(vis, state)
        if self.training:
            Neck_fq, _, Neck_vis, _, tccl_feature = neck_outputs
        else:
            Neck_fq, _, Neck_vis, tccl_feature = neck_outputs
        b, c, h, w = Neck_fq.size()

        decoder_fq = self.decoder(Neck_fq, word, pad_mask)
        decoder_fq = decoder_fq.reshape(b, c, h, w)
        pred = self.proj(decoder_fq, state)
        features = {
            'vis': vis,  
            'vis_feats': vis_feats, 
            'Neck_fq': Neck_fq,  
            'Neck_vis': Neck_vis,  
            'tccl_feature': tccl_feature,  
            'decoder_fq': decoder_fq, 
            'pred': pred, 
        }

        return pred.detach(), features, None, word, state