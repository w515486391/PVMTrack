"""
Basic vim model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.pvmtrack.models_mamba import create_block
from timm.models import create_model
import torch.nn.functional as F
#from lib.models.osmtrack.mamba_cross import CrossMamba
from thop import profile
from lib.config.pvmtrack.config import cfg

from lib.models.pvmtrack.TPG import TPG
from lib.models.pvmtrack.loss_functions import DJSLoss
from lib.models.pvmtrack.statistics_network import (
    GlobalStatisticsNetwork,
)

from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
class pvmtrack(nn.Module):
    """ This is the base class for vim """

    def __init__(self, visionmamba, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionmamba
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.djs_loss = DJSLoss()
        self.feature_map_size = 8  # 128x128
        self.feature_map_channels = visionmamba.embed_dim
        self.num_ch_coding = self.backbone.embed_dim
        self.coding_size = 8
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.feature_map_size,
            feature_map_channels=self.feature_map_channels,
            coding_channels=self.num_ch_coding,
            coding_size=self.coding_size,
        )

        self.fuse_z = None

        self.TPG = TPG(inplanes=3, hide_channel=24, smooth=True)

    def forward(self, template: torch.Tensor,
                new_template: torch.Tensor,
                search: torch.Tensor,
                template_anno: torch.Tensor,
                search_anno: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        if self.training:
            template_anno = torch.round(template_anno * 8).int()
            template_anno[template_anno < 0] = 0
            search_anno = torch.round(search_anno * 16).int()
            search_anno[search_anno < 0] = 0

        # 模板融合
        b, c, h, w = new_template.shape
        self.fuse_z = self.TPG(torch.cat([template, new_template], dim=1)).contiguous()


        # def visualize_and_save_features(features, feature_names, save_dir):
        #     os.makedirs(save_dir, exist_ok=True)
        #     for feature, name in zip(features, feature_names):

        # feature = self.fuse_z.squeeze(0)  # Reduce batch dimension
        # mean_feature = torch.mean(feature, dim=0).detach().cpu().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(mean_feature, cmap='viridis')
        # plt.axis('off')
        # save_path = os.path.join(f'/media/yy/4T/wbb4/pvmtrack-main/output', f"debug_TPG.png")
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()


        x_24 = self.backbone.forward_features( z=template, zd=self.fuse_z, x=search,
                                                inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        x_26 = self.backbone.forward_features1( z=template, zd=self.fuse_z, x=search, x24=x_24,
                                                inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        x_28 = self.backbone.forward_features2( z=template, zd=self.fuse_z, x=search, x26=x_26,
                                                inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        # Forward head
        # search_feature = x[:, -self.feat_len_s:]
        # x = search_feature
        # feat_last = search_feature

        feat_last = x_24
        feat_last_1 = x_26
        feat_last_2 = x_28

        # if isinstance(x, list):
        #     feat_last = x[-1]
        out = self.forward_head(feat_last, feat_last_1, feat_last_2, None, template_anno=template_anno, search_anno=search_anno)
       
        out['backbone_feat'] = x_24
        return out

    def forward_head(self, cat_feature, cat_feature_1, cat_feature_2, gt_score_map=None, template_anno=None, search_anno=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        #MI
        if self.training:

            mi = cat_feature[:, -320:, :]

            feat_len_t = mi.shape[1] - self.feat_len_s
            feat_sz_t = int(math.sqrt(feat_len_t))
            enc_opt_z = mi[:, 0:feat_len_t]
            opt = (enc_opt_z.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat_z = opt.view(-1, C, feat_sz_t, feat_sz_t)
        #MI

        search_feature = cat_feature[:, -self.feat_len_s:]
        opt = (search_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        search_feature_1 = cat_feature_1[:, -self.feat_len_s:]
        opt_1 = (search_feature_1.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt_1.size()
        opt_feat_26 = opt_1.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        search_feature_2 = cat_feature_2[:, -self.feat_len_s:]
        opt_2 = (search_feature_2.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt_2.size()
        opt_feat_28 = opt_2.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        opt_feat_ = opt_feat_28 + opt_feat_26
        opt_feat__ = F.relu(opt_feat_)
        opt_feat_l = opt_feat + opt_feat__

        # def visualize_and_save_features(features, feature_names, save_dir):
        #     os.makedirs(save_dir, exist_ok=True)
        #     for feature, name in zip(features, feature_names):

        # feature = opt_feat_28.squeeze(0)  # Reduce batch dimension
        # mean_feature = torch.mean(feature, dim=0).detach().cpu().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(mean_feature, cmap='viridis')
        # plt.axis('off')
        # save_path = os.path.join(f'/media/yy/4T/wbb4/pvmtrack-main/output', f"debug.png")
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()
        #
        # feature_l = opt_feat_l.squeeze(0)  # Reduce batch dimension
        # mean_feature_l = torch.mean(feature_l, dim=0).detach().cpu().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(mean_feature_l, cmap='viridis')
        # plt.axis('off')
        # save_path = os.path.join(f'/media/yy/4T/wbb4/pvmtrack-main/output', f"debug_l.png")
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()

        global_mutual_loss = torch.zeros(0)
        if self.training:
            opt_feat_mask = torch.zeros(mi.shape[0], mi.shape[2], 8, 8)
            opt_feat_x = torch.zeros(mi.shape[0], mi.shape[2], 8, 8)

            template_anno_ = template_anno[0].unsqueeze(0).repeat(4, 1, 1)

            for i in range(opt_feat.shape[0]):
                # if template_anno.shape[0]==1:
                #     bbox = template_anno.squeeze()[i]
                #     bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])
                #     x_t = bbox[0]
                #     y_t = bbox[1]
                # # 默认样本数量为2
                # else:
                #     bbox = template_anno[0][i]
                #     bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])
                #     x_t = bbox[0]
                #     y_t = bbox[1]

                # bbox = template_anno.squeeze()[i]
                bbox = template_anno_.flatten(0 ,1)[i]

                bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])
                x_t = bbox[0]
                y_t = bbox[1]

                target_sz_t = opt_feat_mask[i, :, y_t:y_t + bbox[3], x_t:x_t + bbox[2]].shape

                # bbox = search_anno.squeeze()[i]
                bbox = search_anno.flatten(0 ,1)[i]
                bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])

                target_sz_s = opt_feat[i, :, bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]].shape
                h = min([target_sz_t[1], target_sz_s[1]])
                w = min([target_sz_t[2], target_sz_s[2]])
                opt_feat_x[i, :, y_t:y_t + h, x_t:x_t + w] = opt_feat[i, :, bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w]
                opt_feat_mask[i, :, y_t:y_t + h, x_t:x_t + w] = 1

            opt_feat_z = opt_feat_z * opt_feat_mask.to(opt_feat_z.device)

            x = opt_feat_z.to(opt_feat.device)
            y = opt_feat_x.to(opt_feat.device)
            x_shuffled = torch.cat([x[1:], x[0].unsqueeze(0)], dim=0)

            # Global mutual information estimation
            global_mutual_M_R_x = self.global_stat_x(x, y)  # positive statistic
            global_mutual_M_R_x_prime = self.global_stat_x(x_shuffled, y)
            global_mutual_loss = self.djs_loss(
                T=global_mutual_M_R_x,
                T_prime=global_mutual_M_R_x_prime,
            )


        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
           
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat_l, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'mine_loss': global_mutual_loss,
                   }
            return out
        else:
            raise NotImplementedError


def build_pvmtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('pvmtrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    backbone = create_model( model_name= cfg.MODEL.BACKBONE.TYPE, pretrained= pretrained, num_classes=1000,
            drop_rate=0.0, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, drop_block_rate=None, img_size=256
            )
    hidden_dim = 384
    box_head = build_box_head(cfg, hidden_dim)
    model = pvmtrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
   
    if 'OSMTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    #print(1/0)
    return model

if __name__ == '__main__':
    net = build_pvmtrack(cfg)
    net = net.cuda()
    var1 = torch.Tensor(1, 3, 128, 128).cuda()
    var2 = torch.Tensor(1, 3, 256, 256).cuda()

    out = net(var1, var2)
    print("over")

   
