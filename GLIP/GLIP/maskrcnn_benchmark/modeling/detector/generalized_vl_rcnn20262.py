# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""
import os.path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.utils import cat, permute_and_flatten
from ..backbone import build_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads

from ..language_backbone import build_language_backbone
from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy
import numpy
import matplotlib.pyplot as plt
import numpy as np
def get_distinct_colors(n,seed=42):
    np.random.seed(seed)
    cmap = plt.get_cmap('hsv')
    # 生成 n 个等间距的颜色
    colors = [cmap(i / (n - 1)) for i in range(n)]
    # np.random.shuffle(colors)
    # 将颜色转换为 RGB 格式
    rgb_colors = [(color[0], color[1], color[2]) for color in colors]
    return rgb_colors
def cos_similarity(tensorx):
    sim_=torch.matmul(tensorx,tensorx.transpose(-1,-2)).cpu().detach().numpy()
    sim_range01=sim_.copy()
    for i in range(sim_.shape[-2]):
        for j in range(sim_.shape[-1]):
            sim_range01[i,j]/=numpy.sqrt(sim_[i,i])
            sim_range01[i, j] /= numpy.sqrt(sim_[j, j])
    return sim_range01
def cos_similarity2(tensorx,tensorb):
    sim_=torch.matmul(tensorx,tensorb.transpose(-1,-2)).cpu().detach().numpy()
    sim_range01=sim_.copy()
    # for i in range(sim_.shape[-2]):
    #     for j in range(sim_.shape[-1]):
    #         sim_range01[i,j]/=numpy.sqrt(sim_[i,i])
    #         sim_range01[i, j] /= numpy.sqrt(sim_[j, j])
    return sim_range01
def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j, i] == -1:
                output_label[j, i] = -100
                continue

            if (not input_ids[j, i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j, i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j, i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j, i] = -100

            if greenlight_map is not None and greenlight_map[j, i] != 1:
                output_label[j, i] = -100  # If this location should not be masked
    return input_ids, output_label

def embed_some_768_to_lang_embedding_right_positions_DECREPTED(new_768s_to_plug_in, reference_map, positive_map,
                                                     position_of_lang_embedding):  # reference_map can provide the right position
    plist = position_of_lang_embedding.tolist()
    labels = reference_map[0].extra_fields['labels']
    if self.training:  # in train phase, positive_map is 0-1value few-hot mask liked 256-vector
        plist[0].insert(positive_map.argmax(dim=1) + 1, new_768s_to_plug_in)
        positive_map_list = positive_map.tolist()
        positive_map_list[0].insert(positive_map.argmax(dim=1) + 1, 1)
        positive_map = torch.tensor(positive_map_list[0][:256]).unsqueeze(0)
    else:  # in test phase ,positive_map is dict of indexes, such as {1:[1],2:[3,4],3:[5,6,7]} for "person. star fish. boys and girls."
        tar_dict_key = positive_map[int(labels[0])]
        plist[0].insert(tar_dict_key[-1] + 1, new_768s_to_plug_in)
        tar_dict_key.append(tar_dict_key[-1] + 1)
        positive_map.update({int(labels[0]): tar_dict_key})
    plist[0] = plist[0][:256]
    position_of_lang_embedding = torch.tensor(plist).cuda()
    return positive_map, position_of_lang_embedding
def regenerate_positive_map(tokenizer_input, positive_map, appending=0,feats=[1,1,1,1,1,1,1,1,1,1,1,1]):
    positive_map.clear()
    input_ids = tokenizer_input['input_ids'][tokenizer_input['input_ids'] != 0]
    phrase_count = 0
    positions_list_of_current_phrase = []
    positions_list_of_DOT=[]
    CURRENT_position_inserted = False
    for id, t_num in enumerate(input_ids):
        if phrase_count >= 79:
            break
        if t_num == 101:  # 跳过CLS token
            continue
        elif t_num == 1012 or t_num == 102:  # 遇到句点/SEP，保存当前phrase
            if appending >= 1 and feats:
                positions_list_of_DOT.append(id)
                if feats[phrase_count] is not None:
                    CURRENT_position_inserted=True
            if len(positions_list_of_current_phrase) > 0:
                phrase_count += 1
                # 新增：appending=1时，追加最后一个位置+1
                if appending >= 1 and positions_list_of_current_phrase and CURRENT_position_inserted:
                    last_pos = positions_list_of_current_phrase[-1]
                    positions_list_of_current_phrase.append(last_pos + 1)
                # 原逻辑：更新positive_map
                positive_map.update({phrase_count: positions_list_of_current_phrase})
                positions_list_of_current_phrase = []
        else:
            positions_list_of_current_phrase.append(id)
    return positive_map,positions_list_of_DOT
class conv_map_module(nn.Module):
    """ map support image feature (5-stage) to text space."""

    def __init__(self, in_channels, out_channels, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.drop = nn.Dropout(drop)
        self.fc=nn.Linear(5*in_channels*7*7,768)
        self.act = act_layer(inplace=True)
    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv2(self.conv1(x[0]))))
        x2 = self.conv4(self.conv3(self.conv2(x[1])))
        x3 = self.conv4(self.conv3(x[2]))
        x4 = self.conv4(x[3])
        x5 = x[4]
        x=cat((x1,x2,x3,x4,x5),dim=0)
        x=self.fc(x.flatten())
        # x = self.act(x)
        # x = self.drop(x)
        return x
class conv_map_module0(nn.Module):
    """ map support image feature (5-stage) to text space. Ablation study version """

    def __init__(self, in_channels, out_channels, out_features=None, act_layer=nn.ReLU, drop=0., share=0, scale_num=5):
        super().__init__()
        self.share=share
        self.scale_num=scale_num
        for i in range(self.scale_num):
            for stage in range(4-i):
                exec('self.conv{}{} = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))'.format(i,stage))
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.drop = nn.Dropout(drop)
        self.fc=nn.Linear(scale_num*in_channels*7*7,768)
        self.act = act_layer(inplace=True)
    def forward(self, x):
        for i in range(self.scale_num):
            exec('x{} = x[{}]'.format(i,i))
            for stage in range(4-i):
                exec('x{} = self.conv{}{}(x{})'.format(i,i,stage,i))
        # x0 = self.conv4(self.conv3(self.conv2(self.conv1(x[0]))))
        # x1 = self.conv4(self.conv3(self.conv2(x[1])))
        # x2 = self.conv4(self.conv3(x[2]))
        # x3 = self.conv4(x[3])
        # x4 = x[4]
        cat_tuple=''
        for i in range(self.scale_num):
            cat_tuple=cat_tuple+',x{}'.format(i)
        x=eval('cat(({},),dim=0)'.format(cat_tuple[1:]))
        x=self.fc(x.flatten())
        # x = self.act(x)
        # x = self.drop(x)
        return x


class GeneralizedVLRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        # if cfg.IMPROMPT.gvl:
        #     cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM=13343
        self.cfg = cfg
        # visual encoder
        self.backbone = build_backbone(cfg)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                   from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                   from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg)

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.DEBUG = cfg.MODEL.DEBUG
        if abs(cfg.IMPROMPT.gvl)==1:
            self.reference_image_tensor=None
            self.reference_gt=None
            self.reference_length=0
            self.reference_length_make_sure_same_dataset=0
            self.reference_image_tensor_make_sure_same_dataset = []
            self.history_reference_image_tensor=[0 for x in range(0,80)]
        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER
        if cfg.IMPROMPT.gvl2 >= 1:
            self.feats=[]
        class Mlp(nn.Module):
            """ Multilayer perceptron."""

            def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
                super().__init__()
                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
                self.act = act_layer(inplace=True)
                self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.act(x)  # caution: remember uncomment to rollback
                x = self.drop(x)
                return x
        if self.cfg.generalized_vl:
            self.adapter = []
            for i in range(5):
                self.adapter.append(nn.Conv2d(256, 256, kernel_size=1, stride=1))

            mlp_v2=True# caution: remember change to false to rollback!!!!!!!!!!!!!!!!!!!!
            cin = 256 * 7 * 7
            cout=int(0.5*cin)
            if mlp_v2:
                self.my_fc= nn.Sequential(
                    Mlp(cin,cout,cin),
                )
            else:
                self.my_fc = nn.Sequential(
                    nn.Linear(cin, cout, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(cout, cin, bias=False),
                    nn.ReLU(inplace=True)
                )
        ##############self.my_fc=nn.Linear(4096,4096,bias=False)
        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES
        if cfg.IMPROMPT.gvl==-1:
            if self.cfg.IMPROMPT.map_module == 'mlp':
                self.adaptivepool_of_stages=[]
                self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
                self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
                self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
                self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
                self.adaptivepool_of_stages.append(nn.Identity())
            if cfg.IMPROMPT.map_module=='mlp':
                self.fcto768=Mlp(in_features=49*256,hidden_features=768,out_features=768)
            elif cfg.IMPROMPT.map_module=='conv':
                if cfg.IMPROMPT.conv_scales_weight_sharing_==1:
                    self.fcto768=conv_map_module(256,256)
                else:
                    self.fcto768 = conv_map_module0(256, 256,scale_num=cfg.IMPROMPT.conv_scales_used)
        from einops import rearrange
        class CrossAttention(nn.Module):
            def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
                super(CrossAttention, self).__init__()
                self.emb_dim = emb_dim
                self.scale = emb_dim ** -0.5

                self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

                self.Wq = nn.Linear(emb_dim, emb_dim)
                self.Wk = nn.Linear(emb_dim, emb_dim)
                self.Wv = nn.Linear(emb_dim, emb_dim)

                self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

            def forward(self, x, context, pad_mask=None):
                '''

                :param x: [batch_size, c, h, w]
                :param context: [batch_szie, seq_len, emb_dim]
                :param pad_mask: [batch_size, seq_len, seq_len]
                :return:
                '''
                b, c, h, w = x.shape
                with torch.autograd.set_detect_anomaly(True):
                    x2 = self.proj_in(x)  # [batch_size, c, h, w] = [3, 512, 512, 512]
                    #x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, c] = [3, 262144, 512]
                    x3= x2.permute(0, 2,3,1)
                    x4=x3.view(b,h*w,self.emb_dim)

                    Q = self.Wq(x4)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
                    K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
                    V = self.Wv(context)

                    # [batch_size, h*w, seq_len]
                    att_weights = torch.einsum('bid,bjd -> bij', Q, K)
                    att_weights = att_weights * self.scale

                    if pad_mask is not None:
                        # [batch_size, h*w, seq_len]
                        att_weights = att_weights.masked_fill(pad_mask, -1e9)

                    att_weights = F.softmax(att_weights, dim=-1)
                    out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, h*w, emb_dim]
                    out = out.permute(0,2,1)
                    out=out.view(b,self.emb_dim,h,w)
                    # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
                    out = self.proj_out(out)  # [batch_size, c, h, w]

                # print(out.shape)

                return x, att_weights
        self.vl_cross_att=cfg.vl_cross_att
        if cfg.vl_cross_att or cfg.apoadapter:
            print('vl_cross_att equipped!!!!!!!!!!!')
            self.cross_attention1= CrossAttention(256,768)
            self.cross_attention2 = CrossAttention(256, 768)
            self.cross_attention3 = CrossAttention(256, 768)
            self.cross_attention4 = CrossAttention(256, 768)
            self.cross_attention5 = CrossAttention(256, 768)
        if cfg.apoadapter:
            self.cross_attention_l2v=CrossAttention(768,256)
        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, 'fpn'):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, 'cls_logits'):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False

        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if self.cfg.GLIPKNOW.KNOWLEDGE_FILE:
            from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
            self.class_name_to_knowledge = load_from_yaml_file(self.cfg.GLIPKNOW.KNOWLEDGE_FILE)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN, self).train(mode)
        if self.freeze_backbone:
            self.backbone.body.eval()
            for p in self.backbone.body.parameters():
                p.requires_grad = False
        if self.freeze_fpn:
            self.backbone.fpn.eval()
            for p in self.backbone.fpn.parameters():
                p.requires_grad = False
        if self.freeze_rpn:
            if hasattr(self.rpn, 'head'):
                self.rpn.head.eval()
            for p in self.rpn.parameters():
                p.requires_grad = False
        if self.linear_prob:
            if self.rpn is not None:
                for key, value in self.rpn.named_parameters():
                    if not (
                            'bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
            if self.roi_heads is not None:
                for key, value in self.roi_heads.named_parameters():
                    if not (
                            'bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
        if self.freeze_cls_logits:
            if hasattr(self.rpn.head, 'cls_logits'):
                self.rpn.head.cls_logits.eval()
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False
        if self.add_linear_layer:
            if self.rpn is not None:
                for key, p in self.rpn.named_parameters():
                    if 'tunable_linear' in key:
                        p.requires_grad = True

        if self.freeze_language_backbone:
            self.language_backbone.eval()
            for p in self.language_backbone.parameters():
                p.requires_grad = False
    def embed_default_word_to_captions_at_GT_position(self,captions,targets,positive_map,reference_map):
        default_word=' photo '
        if self.training:  # in train phase, positive_map is 0-1value few-hot mask liked 256-vector
            label = targets[0].extra_fields['labels'][0]
            pos_indexes=reference_map[label]
            if positive_map[pos_indexes]!=1:
                print('bug!!!!!!!!!!!!!!!!!')
            else:
                positive_map=torch.cat((positive_map[:pos_indexes[-1]+1],torch.ones((1,1)),positive_map[pos_indexes[-1]+1:]),dim=1)
            #TODO: change all_posi;;;;
            p_joint_place=torch.tensor(positive_map.argmax(dim=1))
        return
    def embed_some_768_to_lang_embedding_right_positions(self,new_768s_to_plug_in, reference_map, positive_map,
                                                         position_of_lang_embedding,shot_num=1,shot_fusion=0,stage_num=0,stage_fusion=0):  # reference_map can provide the right position
        labels = reference_map[0].extra_fields['labels']
        if stage_fusion!=shot_fusion:
            shot_fusion=stage_fusion
        if shot_fusion=='average' and shot_num!=1:
            # new_768s_to_plug_in=new_768s_to_plug_in.mean(dim=0)
            COUNT=new_768s_to_plug_in[0]
            for i in range(1,len(new_768s_to_plug_in)):
                COUNT=torch.add(COUNT,new_768s_to_plug_in[i])
            COUNT/=len(new_768s_to_plug_in)
            new_768s_to_plug_in=COUNT
        if shot_fusion=='add' and shot_num!=1:
            # new_768s_to_plug_in=new_768s_to_plug_in.mean(dim=0)
            COUNT=new_768s_to_plug_in[0]
            for i in range(1,len(new_768s_to_plug_in)):
                COUNT=torch.add(COUNT,new_768s_to_plug_in[i])
            new_768s_to_plug_in=COUNT
        if shot_fusion=='max' and shot_num!=1:
            if isinstance(new_768s_to_plug_in,list):
                new_768s_to_plug_in = torch.stack(new_768s_to_plug_in, 0)
                new_768s_to_plug_in = torch.max(new_768s_to_plug_in,0).values
        if shot_fusion=='residual' and shot_num!=1:
            if isinstance(new_768s_to_plug_in,list):
                new_768s_to_plug_in = torch.stack(new_768s_to_plug_in, 0)
                new_768s_to_plug_in = torch.mean(new_768s_to_plug_in,0).values
        if shot_fusion!='concat':
            shot_num=1
        if self.training:  # in train phase, positive_map is 0-1value few-hot mask liked 256-vector
            # p_joint_place=torch.tensor(positive_map.argmax(dim=1))
            for latei in range(100):
                if positive_map[0,positive_map[0,:].argmax(dim=0)+latei]==0:
                    the_next_dot_place=positive_map[0,:].argmax(dim=0)+latei
                    break
            if self.cfg.IMPROMPT.override_text_prompt:
                the_next_dot_place -= 1
            positive_map[:, the_next_dot_place] = 1
            # if shot_num > 1:
            #     positive_map[:,the_next_dot_place]=1
            #     positive_map[:, -(shot_num - 1):] = 1
            # position_of_lang_embedding=torch.cat((position_of_lang_embedding[:,:positive_map.argmax(dim=1)+1,:],new_768s_to_plug_in,position_of_lang_embedding[:,positive_map.argmax(dim=1)+1:,:]),dim=1)
            # positive_map =torch.cat((positive_map[:,positive_map.argmax(dim=1)+1],torch.ones((1,1)),positive_map[:,positive_map.argmax(dim=1)+1:]),dim=1)
        else:  # in test phase ,positive_map is dict of indexes, such as {1:[1],2:[3,4],3:[6,7,8]} for "person. star fish. boys and girls."
            try:
                tar_dict_key = positive_map[int(labels[0])]
                the_next_dot_place = tar_dict_key[-1] + 1
                if self.cfg.IMPROMPT.override_text_prompt:
                    the_next_dot_place -= 1
                if shot_num >= 1:
                    if the_next_dot_place not in tar_dict_key:
                        tar_dict_key.append(the_next_dot_place)
                    for i in range(shot_num-1):
                        tar_dict_key.append(255-i)
                    positive_map.update({int(labels[0]): tar_dict_key})
            except:
                return positive_map, position_of_lang_embedding
            # position_of_lang_embedding = torch.cat((position_of_lang_embedding[:, :the_next_dot_place, :],
            #                                         position_of_lang_embedding[:, the_next_dot_place-1:the_next_dot_place, :],
            #                                         position_of_lang_embedding[:, the_next_dot_place:, :]),
            #                                        dim=1)[:,:256,:]
            # tar_dict_key.append(tar_dict_key[-1] + 1)
            # positive_map.update({int(labels[0]): tar_dict_key})
            # for i in range(labels[0]+1,len(positive_map)+1):
            #     the_next_dict_key=positive_map[i]
            #     for ind in range(len(the_next_dict_key)):
            #         the_next_dict_key[ind]+=1
        # position_of_lang_embedding[:, the_next_dot_place, :] = position_of_lang_embedding[:, the_next_dot_place - 1, :]
        if shot_num==1 or self.training:
            new_768s_expanded = new_768s_to_plug_in.unsqueeze(0).unsqueeze(0)  # [768] → [1,1,768]
            new_768s_expanded = new_768s_expanded.to(dtype=position_of_lang_embedding.dtype)  # 对齐dtype
            idx = torch.tensor([[[the_next_dot_place]]],
                               dtype=torch.long,
                               device=position_of_lang_embedding.device)
            idx_expanded = idx.expand(-1, -1, 768)  # [1,1,1] → [1,1,768]
            position_of_lang_embedding = position_of_lang_embedding.scatter_(
                dim=1,  # 针对seq_len维度（第1维）替换
                index=idx_expanded,  # 替换位置索引
                src=new_768s_expanded  # 替换值（dtype已对齐）
            )
        else:
            # position_of_lang_embedding[:, the_next_dot_place, :] = new_768s_to_plug_in[0]
            new_768s_expanded = new_768s_to_plug_in[0].unsqueeze(0).unsqueeze(0)  # [768] → [1,1,768]
            new_768s_expanded = new_768s_expanded.to(dtype=position_of_lang_embedding.dtype)  # 对齐dtype
            idx = torch.tensor([[[the_next_dot_place]]],
                               dtype=torch.long,
                               device=position_of_lang_embedding.device)
            idx_expanded = idx.expand(-1, -1, 768)  # [1,1,1] → [1,1,768]
            position_of_lang_embedding = position_of_lang_embedding.scatter_(
                dim=1,  # 针对seq_len维度（第1维）替换
                index=idx_expanded,  # 替换位置索引
                src=new_768s_expanded  # 替换值（dtype已对齐）
            )
            for shot_i in range(len(new_768s_to_plug_in)-1):
                position_of_lang_embedding[:,-shot_i,:]=new_768s_to_plug_in[shot_i+1]
        # position_of_lang_embedding = position_of_lang_embedding[:,:256,:]
        return positive_map, position_of_lang_embedding
    def forward(self,
                images,
                targets=None,
                captions=None,
                positive_map=None,
                greenlight_map=None,
                reference_map=None,
                idx=0
                ):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # if self.cfg.print_flops:
        #     self.training=True
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # if self.cfg.IMPROMPT.gvl == -1:
        #     captions,positive_map=self.embed_default_word_to_captions_at_GT_position(captions,targets,positive_map,reference_map)
        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device
        if self.cfg.print_flops:
            import time
            start_time = time.time()

        language_dict_features = {}
        if captions is not None:
            # print(captions[0])
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                         max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                         padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                         return_special_tokens_mask=True,
                                                         return_tensors='pt',
                                                         truncation=True).to(device)

            input_ids = tokenized.input_ids
            mlm_labels = None
            tokenizer_input = {"input_ids": input_ids,
                               "attention_mask": tokenized.attention_mask}
            if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                if self.cfg.FROZEE_BERT:
                    for name, param in self.language_backbone.named_parameters():
                        if 'adapter' not in name and 'lora_' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    if self.cfg.use_lora_text:
                        import loralib as lora
                        lora.mark_only_lora_as_trainable(self)
                    language_dict_features = self.language_backbone(tokenizer_input)
                else:
                    language_dict_features = self.language_backbone(tokenizer_input)
            else:
                if self.cfg.FROZEE_BERT:
                    for name, param in self.language_backbone.named_parameters():
                        if 'adapter' not in name: ##wuyongjian edit: here is for the text adapter ABC
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    if self.cfg.use_lora_text:
                        import loralib as lora
                        lora.mark_only_lora_as_trainable(self)
                    language_dict_features = self.language_backbone(tokenizer_input)
                else:
                    language_dict_features = self.language_backbone(tokenizer_input)
            if not self.training and positive_map is not None:
                positive_map,positions_list_of_DOT=regenerate_positive_map(tokenizer_input,positive_map)
            language_dict_features["mlm_labels"] = mlm_labels

        swint_feature_c4 = None
        FROZEE_SWINT = self.cfg.FROZEE_SWINT

        if self.cfg.IMPROMPT.gvl2>0 and not self.training:
            if FROZEE_SWINT:
                # with torch.no_grad():  # wyj : add to freeze visual backbone!!!!!!!!!!!!!!!!!!!
                for name, param in self.backbone.named_parameters():
                    if 'adapter' not in name and 'lora_' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        # if 'layers.1' in name :#or 'layers.3' in name:
                        #     # print(name+'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        #     param.requires_grad = True
                        # else:
                        #     param.requires_grad = False
                # if self.cfg.use_lora_visual:
                #     import loralib as lora
                #     lora.mark_only_lora_as_trainable(self)
                if 'vl' in self.cfg.MODEL.SWINT.VERSION:
                    # the backbone only updates the "hidden" field in language_dict_features
                    inputs = {"img": images.tensors, "lang": language_dict_features}
                    visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
                else:
                    visual_features = self.backbone(images.tensors)
            else:
                if 'vl' in self.cfg.MODEL.SWINT.VERSION:
                    # the backbone only updates the "hidden" field in language_dict_features
                    inputs = {"img": images.tensors, "lang": language_dict_features}
                    visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
                else:
                    if self.cfg.use_maple:
                        visual_features = self.backbone((images.tensors, maple_project_out.view(-1, 192, 1, 1)))
                    else:
                        visual_features = self.backbone(images.tensors)
            if self.cfg.print_flops:
                elapsed_time = time.time() - start_time  # 计算耗时
                print(f"程序已运行: {elapsed_time:.2f} 秒")  # 输出到控制台
            # rpn force boxes
            if targets:
                targets = [target.to(device)
                           for target in targets if target is not None]
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets,
                                                                         language_dict_features, positive_map,
                                                                         captions, swint_feature_c4)
            if self.roi_heads:
                if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                    x, result, detector_losses = self.roi_heads(
                        fused_visual_features, proposals, targets,
                        language_dict_features=language_dict_features,
                        positive_map_label_to_token=positive_map if not self.training else None
                    )
                else:
                    x, result, detector_losses = self.roi_heads(
                        visual_features, proposals, targets,
                        language_dict_features=language_dict_features,
                        positive_map_label_to_token=positive_map if not self.training else None
                    )
            else:
                # RPN-only models don't have roi_heads
                x = visual_features
                result = proposals
                detector_losses = {}
            if not self.training:
                import numpy as np
                import torch
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle

                # ===================== 1. 提取BoxList中的关键信息（修复梯度张量问题） =====================
                boxlist = result[0]  # 取出列表中的BoxList实例
                # 关键修复：detach()分离梯度 → cpu() → numpy()
                scores = boxlist.extra_fields['scores'].detach().cpu().numpy()  # 置信度
                bboxes = boxlist.bbox.detach().cpu().numpy()  # 边界框（xyxy格式）
                labels = boxlist.extra_fields['labels'].detach().cpu().numpy()  # 标签
                locs = boxlist.extra_fields['loc'].detach().cpu().numpy()
                stages = boxlist.extra_fields['stage'].detach().cpu().numpy()

                # 1.2 按9个固定类别提取每个类的最高置信度实例（无则为None）
                # 核心修改：固定9个类别列表，确保无论预测结果如何都有9个类的位置
                FIXED_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 数据集的9个类别，按实际顺序调整
                class_results = {}  # 存储每个类的最高置信度实例信息

                # 遍历9个固定类别，找到每个类的最高置信度实例
                for cls in FIXED_CLASSES:
                    # 筛选该类的所有实例索引
                    cls_indices = np.where(labels == cls)[0]
                    if len(cls_indices) == 0:
                        class_results[cls] = None  # 无该类实例，填充None
                        continue

                    # 找到该类中置信度最高的实例索引
                    cls_scores = scores[cls_indices]
                    max_score_idx_in_cls = np.argmax(cls_scores)  # 该类内最高置信度索引
                    max_idx = cls_indices[max_score_idx_in_cls]  # 全局索引

                    # 存储该类最高置信度实例的所有信息
                    class_results[cls] = {
                        "score": scores[max_idx],
                        "bbox": bboxes[max_idx],
                        "label": labels[max_idx],
                        "loc": locs[max_idx],
                        "stage": stages[max_idx],
                        "global_idx": max_idx  # 全局索引，用于后续取特征
                    }

                # 1.3 按9个固定类别整理结果（无则填充None），并提取特征
                class_bboxes = []  # 9个类的bbox（None表示无该类）
                class_scores = []  # 9个类的score（None表示无该类）
                class_labels = []  # 9个类的label（None表示无该类）

                for cls in FIXED_CLASSES:
                    cls_info = class_results[cls]
                    if cls_info is not None:
                        # 该类有实例，提取信息
                        class_bboxes.append(cls_info["bbox"])
                        class_scores.append(cls_info["score"])
                        class_labels.append(cls_info["label"])

                        # 提取该类最高置信度实例的特征（保留原张量处理逻辑）
                        stage_idx = int(np.round(cls_info["stage"]))
                        loc_idx = int(np.round(cls_info["loc"]))
                        feat = fused_visual_features[stage_idx].reshape(1, 256, -1)[:, :, loc_idx].squeeze(0).detach()

                        # 处理特征平均逻辑
                        if len(self.feats) < cls:
                            # 如果列表长度不足，先填充None到指定位置
                            for _ in range(len(self.feats), cls):
                                self.feats.append(None)

                        # 获取原有特征值
                        original_feat = self.feats[cls - 1]
                        if original_feat is None:
                            # 原有值为None，直接使用新特征
                            self.feats[cls - 1] = feat
                        else:
                            # 原有值存在，计算新旧特征的平均值
                            # 确保在同一设备上计算（CPU/GPU），并保持梯度分离状态
                            avg_feat = (original_feat + feat) / 2.0
                            self.feats[cls - 1] = avg_feat.detach()  # 保持detach状态和原逻辑一致
                    else:
                        # 该类无实例，填充None（确保列表长度始终为9）
                        class_bboxes.append(None)
                        class_scores.append(None)
                        class_labels.append(None)
                        if len(self.feats)<cls:
                            self.feats.append(None)
                # ===================== 2. 处理ImageList，还原原图 =====================
                # 2.1 提取并反归一化图片张量（同样修复梯度问题）
                img_tensor = images.tensors[0].detach().cpu()  # 分离梯度+CPU
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean  # 反归一化
                # 2.2 转换为plt可显示的格式
                img = img_tensor.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)  # 防止像素值溢出

                # ===================== 3. 可视化：绘制9个类中存在的最高置信度实例 =====================
                plt.figure(figsize=(12, 8))  # 画布大小
                plt.imshow(img)
                ax = plt.gca()  # 获取坐标轴

                # 为不同类别定义不同颜色（9个类对应9种颜色，方便区分）
                class_colors = {
                    1: 'green', 2: 'red', 3: 'blue', 4: 'yellow', 5: 'purple',
                    6: 'orange', 7: 'cyan', 8: 'magenta', 9: 'brown'
                }

                # 遍历9个固定类别的结果，只绘制有实例的类别
                for cls, bbox, score, label in zip(FIXED_CLASSES, class_bboxes, class_scores, class_labels):
                    if bbox is not None and score is not None and label is not None:
                        # 仅绘制有有效实例的类别
                        x1, y1, x2, y2 = bbox
                        # 选择该类对应的颜色
                        color = class_colors.get(cls, 'gray')  # 兜底颜色为灰色
                        # 绘制矩形框
                        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor=color,
                                         facecolor='none')
                        ax.add_patch(rect)
                        # 绘制文本（带背景，显示类别+置信度）
                        text = f"Class:{label} | Score:{score:.2f}"
                        ax.text(x1, y1 - 5, text,
                                fontsize=10, color='white',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

                # 优化显示效果
                plt.title(f"Top-1 Instance per Class (Total 9 Classes)", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                # plt.show()
                # # 打印日志，确认9个类的结果状态（方便调试）
                # print("===== 9个类别结果状态 =====")
                # for idx, cls in enumerate(FIXED_CLASSES):
                #     status = "有实例" if class_bboxes[idx] is not None else "无实例"
                #     score = class_scores[idx] if class_scores[idx] is not None else "N/A"
                #     print(f"类别 {cls}: {status} | 置信度: {score}")
        if FROZEE_SWINT:
            # with torch.no_grad():  # wyj : add to freeze visual backbone!!!!!!!!!!!!!!!!!!!
            for name,param in self.backbone.named_parameters():
                if 'adapter' not in name and 'lora_' not in name:
                    param.requires_grad=False
                else:
                    param.requires_grad = True
                    # if 'layers.1' in name :#or 'layers.3' in name:
                    #     # print(name+'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #     param.requires_grad = True
                    # else:
                    #     param.requires_grad = False
            # if self.cfg.use_lora_visual:
            #     import loralib as lora
            #     lora.mark_only_lora_as_trainable(self)
            if 'vl' in self.cfg.MODEL.SWINT.VERSION:
                # the backbone only updates the "hidden" field in language_dict_features
                inputs = {"img": images.tensors, "lang": language_dict_features}
                visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
            else:
                visual_features = self.backbone(images.tensors)
        else:
            if 'vl' in self.cfg.MODEL.SWINT.VERSION:
                # the backbone only updates the "hidden" field in language_dict_features
                inputs = {"img": images.tensors, "lang": language_dict_features}
                visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
            else:
                if self.cfg.use_maple:
                    visual_features = self.backbone((images.tensors,maple_project_out.view(-1,192,1,1)))
                else:
                    visual_features = self.backbone(images.tensors)
        if self.cfg.print_flops:
            elapsed_time = time.time() - start_time  # 计算耗时
            print(f"程序已运行: {elapsed_time:.2f} 秒")  # 输出到控制台
        # rpn force boxes
        if targets:
            targets = [target.to(device)
                       for target in targets if target is not None]
        if not self.training and positive_map is not None:
            if self.cfg.IMPROMPT.gvl2>0:
                positive_map, positions_list_of_DOT = regenerate_positive_map(tokenizer_input, positive_map,appending=self.cfg.IMPROMPT.gvl2,feats=self.feats if self.feats else None)
                self.feats.append(positions_list_of_DOT)
            else:
                positive_map, positions_list_of_DOT = regenerate_positive_map(tokenizer_input, positive_map,
                                                                              appending=self.cfg.IMPROMPT.gvl2)
        if self.cfg.IMPROMPT.gvl2>0 and not self.training:
            language_dict_features['hidden']=language_dict_features['embedded']
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets,
                                                                         language_dict_features, positive_map,
                                                                         captions, swint_feature_c4, self.feats)
        else:
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets,
                                                                             language_dict_features, positive_map,
                                                                             captions, swint_feature_c4)
        if self.roi_heads:
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                x, result, detector_losses = self.roi_heads(
                    fused_visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
            else:
                x, result, detector_losses = self.roi_heads(
                    visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
        else:
            # RPN-only models don't have roi_heads
            x = visual_features
            result = proposals
            detector_losses = {}
        if self.cfg.print_flops:
            elapsed_time = time.time() - start_time  # 计算耗时
            print(f"程序已运行: {elapsed_time:.2f} 秒")  # 输出到控制台
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.cfg.use_dept:
                def _forward_logits_similarity(self, text_feats, img_feats):
                    # normalize and calcute cosine similarity
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    logit_scale = self.logit_scale.exp()
                    logits = logit_scale * img_feats @ text_feats.t()
                    return logits

                def _forward_logits_linear_probe(self, text_feats, img_feats, labels):
                    # cwT module
                    if self.film_cfg.LINEAR_PROBE:
                        text_feats = self.film_lp_text(text_feats)
                        img_feats = self.film_lp_img(img_feats)

                    # while new head is similarity head, use similarity forward function
                    if self.lp_cfg.TYPE == 'similarity':
                        return self._forward_logits_similarity(text_feats, img_feats), labels

                    if labels is None:
                        # while inference, forward image features only
                        all_feats = img_feats
                        all_labels = labels
                    else:
                        # while training, image features and text features will be concated to train classifier
                        text_feats = text_feats[labels]
                        all_feats = torch.cat([text_feats, img_feats])
                        all_labels = torch.cat([labels, labels])

                    all_logits = self.linear_probe_proj(all_feats)
                    return all_logits, all_labels

                def _loss(self, logits, labels, logits_lp, labels_lp):
                    # calculate similarity loss and linear loss
                    loss_cls = F.cross_entropy(logits, labels)
                    loss_cls_lp = F.cross_entropy(logits_lp, labels_lp)

                    lp_weight = self.lp_cfg.WEIGHT
                    loss = (1 - lp_weight) * loss_cls + lp_weight * loss_cls_lp
                    return loss
                """ forward function for base classes """
                text_feats, img_feats = language_dict_features,visual_features

                # forward similartiy and linear logits
                logits = self._forward_logits_similarity(text_feats, img_feats)
                logits_lp, labels_lp = self._forward_logits_linear_probe(text_feats, img_feats, labels)

                if self.training:
                    # while training, return loss of both logits
                    return self._loss(logits, labels, logits_lp, labels_lp)

                if not self.lp_cfg.TEST_TIME_FUSION:
                    return logits_lp

                # while inference, fusion both logits and return
                lp_weight = self.lp_cfg.WEIGHT
                logits = (1 - lp_weight) * logits + lp_weight * logits_lp
                return logits
            return losses
        if self.cfg.IMPROMPT.gvl2:
            # ===================== 核心修改：两列对比可视化 =====================
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            from matplotlib.patches import Rectangle

            # 1. 预处理原图（反归一化）
            img_tensor = images.tensors[0].detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img_tensor * std + mean).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)  # 确保像素值在0-1之间

            # 2. 提取预测结果（置信度>0.5）
            pred_scores = result[0].extra_fields['scores'].detach().cpu().numpy()
            pred_bboxes = result[0].bbox.detach().cpu().numpy()
            pred_labels = result[0].extra_fields['labels'].detach().cpu().numpy()
            keep = pred_scores > 0.5
            pred_bboxes, pred_scores, pred_labels = pred_bboxes[keep], pred_scores[keep], pred_labels[keep]

            # 3. 提取GT（处理targets为空的情况）
            has_gt = False
            gt_bboxes, gt_labels = None, None
            if targets is not None and len(targets) > 0 and targets[0] is not None:
                has_gt = True
                gt_bboxes = targets[0].bbox.detach().cpu().numpy()  # GT边界框
                gt_labels = targets[0].extra_fields['labels'].detach().cpu().numpy()  # GT标签
                # 可选：提取GT masks（如需显示mask可取消注释）
                # gt_masks = targets[0].extra_fields['masks'].detach().cpu().numpy()

            # 4. 创建两列对比的画布
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 1行2列布局

            # ===== 左列：绘制预测结果 =====
            ax1.imshow(img)
            ax1.set_title('Prediction (Conf > 0.5)', fontsize=14, pad=10)
            ax1.axis('off')
            # 绘制预测框
            for bbox, score, label in zip(pred_bboxes, pred_scores, pred_labels):
                if  label==3:
                    x1, y1, x2, y2 = bbox
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green', facecolor='none')
                    ax1.add_patch(rect)
                    text = f"Label:{label} | Score:{score:.2f}"
                    ax1.text(x1, y1 - 5, text, fontsize=9, color='black',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8))

            # ===== 右列：绘制GT标注 =====
            ax2.imshow(img)
            if has_gt:
                ax2.set_title('Ground Truth (GT)', fontsize=14, pad=10)
                # 绘制GT框
                for bbox, label in zip(gt_bboxes, gt_labels):
                    x1, y1, x2, y2 = bbox
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
                    ax2.add_patch(rect)
                    text = f"Label:{label} (GT)"
                    ax2.text(x1, y1 - 5, text, fontsize=9, color='black',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
                # 可选：绘制GT mask（如需显示取消注释）
                # for mask in gt_masks:
                #     ax2.imshow(mask, alpha=0.3, cmap='Reds')
            else:
                ax2.set_title('Ground Truth (No GT Data)', fontsize=14, pad=10, color='gray')
            ax2.axis('off')

            # 5. 调整布局并显示
            plt.tight_layout()
            # plt.show()
        return result

    def _forward_language_parallel(self, captions=None, targets=None,
                                   device=None, positive_map=None):
        ktype = self.cfg.GLIPKNOW.KNOWLEDGE_TYPE

        def _construct_captions_from_class_names(class_names):
            captions = []
            for c in class_names:
                try:
                    info = self.class_name_to_knowledge[c]
                    cap = info['clean_name']

                    # combine wiki and gpt3 knowledge
                    if self.cfg.GLIPKNOW.WIKI_AND_GPT3:
                        ktype = 'def_wiki'
                        know_seq = info[ktype]

                        ktype = 'gpt3'
                        if ktype == 'gpt3' or type(info[ktype]) == list:
                            know_seq += ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM]])

                        cap += ': ' + know_seq

                    # only one knoweldge source is used
                    else:
                        if ktype and ktype in info and info[ktype]:
                            if ktype == 'gpt3' or type(info[ktype]) == list:
                                know_seq = ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM]])
                            else:
                                know_seq = info[ktype]
                            cap += ': ' + know_seq
                except:
                    cap = c
                    print(f'cap {cap}, c {c}')

                captions.append(cap)
            return captions

        if self.training:
            assert captions is None
            assert targets is not None

            max_classes_per_batch = self.cfg.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN
            if max_classes_per_batch >= len(self.class_name_list):
                shuffled_class_names = self.class_name_list.copy()
                random.shuffle(shuffled_class_names)
                if max_classes_per_batch > len(shuffled_class_names):
                    shuffled_class_names.extend(shuffled_class_names[:max_classes_per_batch
                                                                      - len(shuffled_class_names)])
                    random.shuffle(shuffled_class_names)
            else:
                label_list = []
                label_to_idx = {}
                for target_per_im in targets:
                    labels_per_im = target_per_im.get_field('label_names')
                    for label in labels_per_im:
                        if label not in label_to_idx:
                            label_to_idx[label] = len(label_list)
                            label_list.append(label)

                label_list = label_list[:max_classes_per_batch]
                if len(label_list) < max_classes_per_batch:
                    all_neg_classes = [c for c in self.class_name_list if c not
                                       in label_to_idx]
                    neg_label_list = random.sample(all_neg_classes,
                                                   max_classes_per_batch - len(label_list))
                    label_list.extend(neg_label_list)
                random.shuffle(label_list)
                shuffled_class_names = label_list

            label_to_shuffled_idx = {l: i for i, l in
                                     enumerate(shuffled_class_names)}
            total_boxes = sum(len(t) for t in targets)
            positive_map = torch.zeros((total_boxes, max_classes_per_batch + 1),
                                       device=device)
            offset = 0
            for target_per_im in targets:
                labels_per_im = target_per_im.get_field('label_names')
                for label in labels_per_im:
                    j = label_to_shuffled_idx.get(label, -1)
                    if j >= 0:
                        positive_map[offset, j] = 1
                    offset += 1
            captions = _construct_captions_from_class_names(shuffled_class_names)
            captions.append('')  # onobj at the end, onedet/modeling/rpn/loss.py:719
            batch_size = len(targets)

        else:
            assert captions is not None
            batch_size = 1
            assert len(captions) == 1
            class_names = captions[0]
            max_classes_per_batch = len(class_names)
            captions = _construct_captions_from_class_names(class_names)
            captions.append('')  # onobj at the end, onedet/modeling/rpn/loss.py:719

        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                     max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                     padding="longest",
                                                     return_special_tokens_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True).to(device)
        assert not self.use_mlm_loss
        tokenizer_input = {"input_ids": tokenized.input_ids,
                           "attention_mask": tokenized.attention_mask}

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
        else:
            language_dict_features = self.language_backbone(tokenizer_input)

        assert not self.cfg.DATASETS.ONE_HOT
        assert not self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL

        agg_type = self.cfg.GLIPKNOW.LAN_FEATURE_AGG_TYPE
        agg_feats = language_dict_features['hidden']
        agg_emb = language_dict_features['embedded']
        if agg_type == 'first':
            agg_feats = agg_feats[:, 0, :]
            agg_emb = agg_emb[:, 0, :]
        elif agg_type == 'mean':
            attn_mask = language_dict_features['masks']
            seq_len = attn_mask.sum(-1).unsqueeze(-1).float()
            agg_feats = agg_feats * attn_mask.unsqueeze(-1).float()
            agg_feats = agg_feats.sum(1) / seq_len
            agg_emb = agg_emb * attn_mask.unsqueeze(-1).float()
            agg_emb = agg_emb.sum(1) / seq_len
        else:
            raise ValueError('not supported GLIPKNOW.LAN_FEATURE_AGG_TYPE: {}'.format(agg_type))

        expanded_features = agg_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        expanded_embedding = agg_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        lang_dict = {}
        lang_dict["mlm_labels"] = None
        lang_dict["aggregate"] = None
        lang_dict["embedded"] = expanded_embedding
        lang_dict['hidden'] = expanded_features
        lang_dict["masks"] = torch.ones((batch_size, max_classes_per_batch + 1),
                                        device=device, dtype=language_dict_features['masks'].dtype)
        # in GLIP setting, the token at the end of seqence is usually [PAD], and is masked out
        # if [noobj] is not masked out, the loss sum is very big, as most
        # anchors are matched to [noobj]
        lang_dict["masks"][:, -1] = 0
        return lang_dict, positive_map

